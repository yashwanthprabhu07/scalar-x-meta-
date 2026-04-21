# ============================================================
# training/rollout.py — Run one episode against the OpenEnv server
#
# Uses the WebSocket-based EnvClient (our EnterpriseWorkflowEnv from
# openenv/enterprise_workflow_env/client.py) so episode state is
# maintained across steps. A bare HTTP call creates a fresh env per
# request, which is why sparse-reward termination never fires with
# plain requests — that was the bug in the first version.
#
# TRL's GRPOTrainer uses the same WebSocket protocol, so this is
# the correct shape for training.
# ============================================================

import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional

# Add project root + openenv dir to path so we can import the client
_THIS = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_THIS)
_OPENENV_DIR = os.path.join(_PROJECT, "openenv")
for p in (_PROJECT, _OPENENV_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from enterprise_workflow_env.client import EnterpriseWorkflowEnv
from enterprise_workflow_env.models import EnterpriseAction


# Type for the agent callable
AgentFn = Callable[[str, List[Dict], List[Dict]], Optional[Dict[str, Any]]]


# ─────────────────────────────────────────────
# TOOL SCHEMAS (for the LLM agent to know what's available)
# ─────────────────────────────────────────────
def load_tool_schemas() -> List[Dict]:
    """Load tool definitions from project-root tools.py in OpenAI format."""
    from tools import ALL_TOOLS  # imports from project root (added to sys.path above)
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in ALL_TOOLS
    ]


# ─────────────────────────────────────────────
# CORE ROLLOUT
# ─────────────────────────────────────────────
def run_episode(
    agent_fn: AgentFn,
    base_url: str = "http://localhost:8000",
    max_steps: int = 25,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run one episode against the OpenEnv server.
    Uses the WebSocket-based EnvClient to maintain episode state
    across steps.

    Args:
        agent_fn: callable (system_prompt, conversation, tool_schemas) -> {tool_name, tool_args} | None
        base_url: OpenEnv server URL (default: localhost:8000)
        max_steps: safety cap on agent steps per episode
        verbose: print each tool call as it happens
    """
    tool_schemas = load_tool_schemas()

    # Open a WebSocket session to the env.
    # The client uses the context manager protocol for clean shutdown.
    with EnterpriseWorkflowEnv(base_url=base_url).sync() as client:
        # Reset starts a new episode
        reset_result = client.reset()
        reset_obs = reset_result.observation

        scenario_id = reset_obs.scenario_id
        scenario_info = json.loads(reset_obs.tool_result) if reset_obs.tool_result else {}
        agent_prompt = scenario_info.get("agent_prompt", "")
        required_actions = scenario_info.get("required_actions", [])

        if verbose:
            print(f"── Episode start — scenario: {scenario_id}")
            print(f"   Required actions: {required_actions}")

        system_prompt = (
            "You are an efficient AI employee. You have access to 5 company apps: "
            "Email, Chat, CRM, Tasks, and Calendar. Complete all steps in the user's "
            "task using the available tools. Do not repeat tool calls unnecessarily. "
            "If a tool returns an error, try a different approach."
        )
        conversation = [{"role": "user", "content": agent_prompt}]

        steps: List[Dict] = []
        taken_actions: List[str] = []
        terminated_by = "max_steps"
        final_reward = 0.0
        task_success: Optional[bool] = None
        breakdown: List[str] = []

        for step_idx in range(max_steps):
            # Ask the agent what to do next
            try:
                tool_call = agent_fn(system_prompt, conversation, tool_schemas)
            except Exception as e:
                if verbose:
                    print(f"   ❌ Agent errored at step {step_idx}: {e}")
                terminated_by = "error"
                break

            if tool_call is None or tool_call.get("tool_name") is None:
                if verbose:
                    print(f"   ✋ Agent stopped at step {step_idx}")
                terminated_by = "agent_stop"
                break

            tool_name = tool_call["tool_name"]
            tool_args = tool_call.get("tool_args", {})

            if verbose:
                print(f"   [{step_idx+1}] {tool_name}({json.dumps(tool_args)[:80]})")

            # Send the action to the env (over WebSocket — session persists)
            action = EnterpriseAction(tool_name=tool_name, tool_args=tool_args)
            try:
                step_result = client.step(action)
            except Exception as e:
                if verbose:
                    print(f"   ❌ Env step errored: {e}")
                terminated_by = "error"
                break

            obs = step_result.observation
            reward = step_result.reward or 0.0
            done = step_result.done

            step_record = {
                "tool_name": tool_name,
                "tool_args": tool_args,
                "tool_result": (obs.tool_result or "")[:500],
                "error": obs.error,
                "reward": reward,
                "done": done,
            }
            steps.append(step_record)
            taken_actions.append(tool_name)

            # Thread the tool result back into the conversation
            conversation.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": f"call_{step_idx}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_args),
                    },
                }],
            })
            conversation.append({
                "role": "tool",
                "tool_call_id": f"call_{step_idx}",
                "content": obs.tool_result or obs.error or "",
            })

            if done:
                final_reward = reward
                task_success = obs.task_success
                breakdown = list(obs.breakdown or [])
                terminated_by = "done"
                if verbose:
                    print(f"   🏁 Episode done: reward={reward}, success={task_success}")
                break

        return {
            "scenario_id": scenario_id,
            "agent_prompt": agent_prompt,
            "required_actions": required_actions,
            "steps": steps,
            "taken_actions": taken_actions,
            "final_reward": final_reward,
            "task_success": task_success,
            "breakdown": breakdown,
            "terminated_by": terminated_by,
        }


# ─────────────────────────────────────────────
# SCRIPTED AGENT FOR TESTING (no LLM needed)
# ─────────────────────────────────────────────
def make_scripted_agent(tool_sequence: List[Dict]) -> AgentFn:
    remaining = list(tool_sequence)

    def _agent(system_prompt, conversation, tool_schemas):
        if not remaining:
            return None
        return remaining.pop(0)

    return _agent


# ─────────────────────────────────────────────
# CLI SMOKE TEST
# ─────────────────────────────────────────────
def _smoke_test():
    print("Smoke-testing rollout against http://localhost:8000 (WebSocket) ...")

    scripted = [
        {"tool_name": "read_inbox", "tool_args": {}},
        {"tool_name": "read_email", "tool_args": {"email_id": "email_001"}},
        {"tool_name": "get_deal", "tool_args": {"deal_id": "deal_001"}},
        {"tool_name": "reply_email", "tool_args": {
            "email_id": "email_001",
            "body": "We value your business. Here is a 10% discount to keep you on board.",
        }},
        {"tool_name": "update_deal_stage", "tool_args": {
            "deal_id": "deal_001", "new_stage": "Negotiation",
        }},
        {"tool_name": "add_note", "tool_args": {
            "deal_id": "deal_001", "note": "Offered 10% discount to retain the client.",
        }},
        {"tool_name": "book_meeting", "tool_args": {
            "title": "Acme Follow-up",
            "attendees": ["rajesh.kumar@acmecorp.com"],
            "date": "2026-04-23", "time": "10:00", "duration_mins": 30,
        }},
        {"tool_name": "post_message", "tool_args": {
            "channel": "sales",
            "message": "Acme Corp cancellation: offered 10% discount, follow-up booked.",
        }},
    ]

    agent = make_scripted_agent(scripted)
    result = run_episode(agent_fn=agent, verbose=True)

    print("\n── Episode result ──")
    print(f"Scenario:       {result['scenario_id']}")
    print(f"Terminated by:  {result['terminated_by']}")
    print(f"Steps taken:    {len(result['steps'])}")
    print(f"Actions:        {result['taken_actions']}")
    print(f"Final reward:   {result['final_reward']}")
    print(f"Task success:   {result['task_success']}")
    print("\nReward breakdown:")
    for line in result["breakdown"]:
        print(f"  {line}")


if __name__ == "__main__":
    _smoke_test()