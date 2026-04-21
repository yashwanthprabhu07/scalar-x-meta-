# ============================================================
# agent.py — The AI Agent Loop (Using Groq API — free & fast)
# This is the brain of the project.
# It sends tasks to Groq AI, gets tool call responses,
# runs the tools on mock apps, and loops until done.
#
# Reliability features:
#   - Fail-fast on missing GROQ_API_KEY with a clear message
#   - Retry Groq API calls with exponential backoff (handles
#     transient 429/500/network errors without killing the demo)
#   - Per-tool try/except so a single tool failure doesn't crash
#     the loop — the agent sees the error and can recover
#   - Iteration-limit detection so we don't silently declare
#     success on a runaway loop
# ============================================================
import os
import json
import time
from groq import Groq                 # Groq — free and fast AI API
from dotenv import load_dotenv

# Load GROQ_API_KEY from .env file
load_dotenv()

from mock_apps import EmailApp, ChatApp, CRMApp, TaskApp, CalendarApp
from tools import ALL_TOOLS
from reward import calculate_reward, log_episode


# ─────────────────────────────────────────────
# CREATE FRESH APPS FOR EACH EPISODE
# ─────────────────────────────────────────────
def create_fresh_apps():
    """Creates brand new instances of all 5 apps. Called at start of every episode."""
    return {
        "email":    EmailApp(),
        "chat":     ChatApp(),
        "crm":      CRMApp(),
        "tasks":    TaskApp(),
        "calendar": CalendarApp(),
    }


# ─────────────────────────────────────────────
# GROQ API CALL — with retry and exponential backoff
# ─────────────────────────────────────────────
def groq_call_with_retry(client, *, model, max_tokens, tools, tool_choice, messages,
                         max_retries: int = 3, base_delay: float = 1.0):
    """
    Call Groq's chat completions with retry on transient failures.

    Retries on ANY exception (rate limits, 5xx, network blips, timeouts).
    Uses exponential backoff: 1s, 2s, 4s between retries.
    Raises the final exception if all retries are exhausted.
    """
    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            return client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                messages=messages,
            )
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))   # 1s, 2s, 4s
                print(f"   ⚠️  Groq call failed (attempt {attempt}/{max_retries}): {type(e).__name__}: {str(e)[:120]}")
                print(f"      Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"   ❌ Groq call failed after {max_retries} attempts: {type(e).__name__}: {e}")

    # Exhausted retries — re-raise the last exception
    raise last_exception


# ─────────────────────────────────────────────
# TOOL ROUTER — runs the right app method
# ─────────────────────────────────────────────
def execute_tool(tool_name: str, tool_input: dict, apps: dict) -> str:
    """
    Given a tool name and inputs from the AI,
    calls the right method on the right app.
    Returns result as a JSON string.

    Wraps tool execution in try/except so an exception in one tool
    doesn't crash the whole agent loop — the error is returned to
    the agent as a tool result, so it can see and recover.
    """
    email    = apps["email"]
    chat     = apps["chat"]
    crm      = apps["crm"]
    tasks    = apps["tasks"]
    calendar = apps["calendar"]

    try:
        if tool_name == "read_inbox":
            result = email.read_inbox()
        elif tool_name == "read_email":
            result = email.read_email(tool_input["email_id"])
        elif tool_name == "send_email":
            result = email.send_email(tool_input["to"], tool_input["subject"], tool_input["body"])
        elif tool_name == "reply_email":
            result = email.reply_email(tool_input["email_id"], tool_input["body"])

        elif tool_name == "list_channels":
            result = chat.list_channels()
        elif tool_name == "read_channel":
            result = chat.read_channel(tool_input["channel"])
        elif tool_name == "post_message":
            result = chat.post_message(tool_input["channel"], tool_input["message"])

        elif tool_name == "get_deal":
            result = crm.get_deal(tool_input["deal_id"])
        elif tool_name == "update_deal_stage":
            result = crm.update_deal_stage(tool_input["deal_id"], tool_input["new_stage"])
        elif tool_name == "add_note":
            result = crm.add_note(tool_input["deal_id"], tool_input["note"])
        elif tool_name == "get_contact":
            result = crm.get_contact(tool_input["email"])
        elif tool_name == "create_contact":
            result = crm.create_contact(
                tool_input["name"], tool_input["email"],
                tool_input["company"], tool_input.get("phone", "")
            )

        elif tool_name == "list_tasks":
            result = tasks.list_tasks()
        elif tool_name == "get_task":
            result = tasks.get_task(tool_input["task_id"])
        elif tool_name == "create_task":
            result = tasks.create_task(
                tool_input["title"], tool_input["assigned_to"],
                tool_input.get("priority", "Medium")
            )
        elif tool_name == "assign_task":
            result = tasks.assign_task(tool_input["task_id"], tool_input["user"])
        elif tool_name == "close_task":
            result = tasks.close_task(tool_input["task_id"])

        elif tool_name == "list_meetings":
            result = calendar.list_meetings(tool_input.get("date"))
        elif tool_name == "check_conflicts":
            result = calendar.check_conflicts(
                tool_input["date"], tool_input["time"], tool_input["attendees"]
            )
        elif tool_name == "book_meeting":
            result = calendar.book_meeting(
                tool_input["title"], tool_input["attendees"],
                tool_input["date"], tool_input["time"],
                tool_input.get("duration_mins", 60)
            )
        else:
            result = {"error": f"Unknown tool: {tool_name}"}
    except Exception as e:
        # Return error as tool result so the agent can see it and recover
        result = {"error": f"Tool '{tool_name}' raised an exception: {str(e)}"}

    return json.dumps(result, indent=2)


# ─────────────────────────────────────────────
# MAIN AGENT LOOP
# ─────────────────────────────────────────────
def run_agent(scenario: dict, progress_callback=None):
    """
    Runs the AI agent on one scenario using the Groq API (free & fast).

    Args:
        scenario:           one of the dicts from scenarios.py
        progress_callback:  optional function called after each tool use
                            (used to send live updates to Streamlit)

    Returns:
        dict with episode results (steps, score, success)
    """
    # Fresh apps for this episode
    apps = create_fresh_apps()

    # ── GROQ CLIENT ──────────────────────────────────────────
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Create a .env file in the project root "
            "with a line like:  GROQ_API_KEY=your_key_here\n"
            "You can get a free key at https://console.groq.com"
        )
    client = Groq(api_key=api_key)

    # Track what the agent does
    agent_steps   = []   # Full step details
    taken_actions = []   # Just tool names (for reward calculation)

    # Build tools in OpenAI function-calling format
    grok_tools = [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            }
        }
        for t in ALL_TOOLS
    ]

    # Start the conversation
    messages = [
        {
            "role": "system",
            "content": (
                "You are an efficient AI employee at a company. "
                "You have access to 5 company apps: Email, Chat, CRM, Tasks, and Calendar. "
                "Complete all the steps in the user's task using the available tools. "
                "Be thorough — use every tool needed to fully complete the task. "
                "IMPORTANT: Do not call the same tool more than once unless you have a clear new reason. "
                "Before acting, read the relevant state first (inbox, channel, task). "
                "If a tool returns an error, examine the error message and try a different approach."
            )
        },
        {
            "role": "user",
            "content": scenario["agent_prompt"]
        }
    ]

    print(f"\n{'='*60}")
    print(f"RUNNING SCENARIO: {scenario['name']}")
    print(f"{'='*60}\n")

    max_iterations = 20          # Safety limit — prevents infinite loops
    hit_iteration_limit = False

    # ── THE MAIN LOOP ─────────────────────────────────────────
    for iteration in range(1, max_iterations + 1):
        print(f"[Iteration {iteration}] Asking Groq what to do next...")

        # Ask Groq what to do (it will either call a tool or say it's done).
        # Uses retry-with-backoff so a transient API error doesn't kill the run.
        response = groq_call_with_retry(
            client,
            model="llama-3.3-70b-versatile",     # Best free Groq model for tool use
            max_tokens=4096,
            tools=grok_tools,
            tool_choice="auto",                  # Let Groq decide when to call tools
            messages=messages,
        )

        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason
        print(f"   → Finish reason: {finish_reason}")

        # ── AGENT IS DONE (no more tool calls) ───────────────
        if finish_reason == "stop" or not message.tool_calls:
            final_text = message.content or "Task completed."
            print(f"   ✅ Agent finished: {final_text[:100]}")
            break

        # ── PROCESS TOOL CALLS ───────────────────────────────
        tool_calls = message.tool_calls   # List of tool calls Groq wants to make

        # Add Groq's response (with tool calls) to conversation history
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in tool_calls
            ]
        })

        # Run each tool call and collect results
        for tc in tool_calls:
            tool_name  = tc.function.name
            try:
                tool_input = json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                tool_input = {}
                print(f"   ⚠️  Failed to parse tool arguments for {tool_name}: {e}")

            tool_id    = tc.id

            print(f"   🔧 Tool: {tool_name}({json.dumps(tool_input)[:80]})")

            # Actually run the tool on our mock apps
            tool_result = execute_tool(tool_name, tool_input, apps)
            print(f"   📦 Result: {tool_result[:100]}...")

            # Track this step
            step = {
                "iteration":   iteration,
                "tool_name":   tool_name,
                "tool_input":  tool_input,
                "tool_result": tool_result,
            }
            agent_steps.append(step)
            taken_actions.append(tool_name)

            # Send step to Streamlit dashboard (live feed)
            if progress_callback:
                progress_callback(step)

            # Add tool result back into conversation so Groq can see what happened
            messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": tool_result,
            })
    else:
        # The `for/else` runs when the loop exits WITHOUT `break`
        # → means we hit max_iterations without the agent saying it was done
        hit_iteration_limit = True
        print(f"   ⚠️  Hit iteration limit ({max_iterations}) without agent stopping")

    # ── EPISODE COMPLETE ─────────────────────────────────────
    # Check 1: were all required tools called?
    required_set = set(scenario["required_actions"])
    taken_set    = set(taken_actions)
    tools_ok     = required_set.issubset(taken_set)

    # Check 2: does the final app state match expectations? (state-based check)
    state_ok = True
    state_reasons = ["(no state check defined for this scenario)"]
    if "success_check" in scenario and callable(scenario["success_check"]):
        try:
            state_ok, state_reasons = scenario["success_check"](apps)
        except Exception as e:
            state_ok = False
            state_reasons = [f"❌ State check raised an exception: {str(e)}"]

    # Check 3: did the agent actually finish (not hit the iteration limit)?
    finished_cleanly = not hit_iteration_limit

    # Overall success = all three must pass
    task_success = tools_ok and state_ok and finished_cleanly

    # Score the episode
    reward_result = calculate_reward(
        required_actions=scenario["required_actions"],
        taken_actions=taken_actions,
        task_success=task_success,
    )

    # Attach the state-check details to the reward result (for the dashboard)
    reward_result["state_check_passed"]  = state_ok
    reward_result["state_check_reasons"] = state_reasons
    reward_result["tools_check_passed"]  = tools_ok
    reward_result["finished_cleanly"]    = finished_cleanly

    # Add state-check info into the breakdown so it shows in the UI
    reward_result["breakdown"].append("")
    reward_result["breakdown"].append("── State check ──")
    for line in state_reasons:
        reward_result["breakdown"].append(line)
    if not finished_cleanly:
        reward_result["breakdown"].append("⚠️  Agent hit the iteration limit without stopping cleanly")

    # Save to history
    episode = log_episode(
        scenario_id=scenario["id"],
        reward_result=reward_result,
        agent_steps=agent_steps,
    )

    print(f"\n{'─'*40}")
    print(f"SCORE: {reward_result['score']}  (normalized {reward_result['normalized_score']})")
    print(f"SUCCESS: {task_success}  (tools_ok={tools_ok}, state_ok={state_ok}, finished_cleanly={finished_cleanly})")
    for line in reward_result["breakdown"]:
        print(f"   {line}")

    return {
        "episode": episode,
        "reward":  reward_result,
        "steps":   agent_steps,
        "apps":    apps,
    }