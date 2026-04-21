# ============================================================
# agent.py — The AI Agent Loop (Using Groq API — free & fast)
# This is the brain of the project.
# It sends tasks to Groq AI, gets tool call responses,
# runs the tools on mock apps, and loops until done.
#
# Reliability features:
#   - Fail-fast on missing GROQ_API_KEY with a clear message
#   - Two-key automatic failover: on rate-limit errors,
#     automatically switches to the backup API key
#   - Retry Groq API calls with exponential backoff (handles
#     transient 429/500/network errors without killing the demo)
#   - Per-tool try/except so a single tool failure doesn't crash
#     the loop — the agent sees the error and can recover
#   - Iteration-limit detection so we don't silently declare
#     success on a runaway loop
#   - Dispatch-dict tool routing (clean, extensible, maintainable)
# ============================================================
import os
import json
import time
from groq import Groq                 # Groq — free and fast AI API
from dotenv import load_dotenv

# Load GROQ_API_KEY and GROQ_API_KEY_BACKUP from .env file
load_dotenv()

from mock_apps import EmailApp, ChatApp, CRMApp, TaskApp, CalendarApp
from tools import ALL_TOOLS
from reward import calculate_reward, log_episode


# ─────────────────────────────────────────────
# GROQ CLIENT POOL — supports multiple keys with automatic failover
# ─────────────────────────────────────────────
class GroqClientPool:
    """
    Manages one or more Groq API keys with automatic failover.

    On a rate-limit error (HTTP 429 / RateLimitError), this pool
    transparently switches to the next available key. Other errors
    are retried on the same key (transient network/server issues).

    Why this design matters: retrying a rate-limited key just gets
    rate-limited again. The Step 5 retry logic alone couldn't help
    when a key was exhausted. Failover is the right fix.
    """

    def __init__(self, api_keys: list):
        if not api_keys:
            raise RuntimeError(
                "No Groq API keys provided. Set at least GROQ_API_KEY in .env.\n"
                "Optionally also set GROQ_API_KEY_BACKUP for automatic failover.\n"
                "Get free keys at https://console.groq.com"
            )
        self.api_keys = api_keys
        self.current_index = 0
        self.clients = [Groq(api_key=k) for k in api_keys]
        # Track which keys are known rate-limited so we don't loop back to them
        self.exhausted = [False] * len(api_keys)
        print(f"   🔑 Groq client pool initialized with {len(api_keys)} key(s)")

    def _current_client(self):
        return self.clients[self.current_index]

    def _mark_exhausted_and_rotate(self):
        """Mark the current key as exhausted and advance to the next live key."""
        self.exhausted[self.current_index] = True
        print(f"   🔄 Key #{self.current_index + 1} exhausted, trying next key...")

        for i in range(len(self.clients)):
            next_idx = (self.current_index + 1 + i) % len(self.clients)
            if not self.exhausted[next_idx]:
                self.current_index = next_idx
                print(f"   ✅ Failed over to key #{self.current_index + 1}")
                return True
        # All keys exhausted
        return False

    def call(self, *, model, max_tokens, tools, tool_choice, messages,
             max_retries_per_key: int = 3, base_delay: float = 1.0):
        """
        Call Groq with automatic failover on rate limits + retry on transient errors.

        Strategy:
          - On rate-limit / 429 / quota errors: immediately fail over to next key.
          - On other errors (network, 500, timeout): retry same key with backoff.
          - If all keys exhausted OR all retries on last key fail: raise.
        """
        last_exception = None

        while True:  # loops on key rotation; retries inside
            client = self._current_client()

            for attempt in range(1, max_retries_per_key + 1):
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
                    err_str = str(e).lower()
                    is_rate_limit = (
                        "rate_limit" in err_str
                        or "rate limit" in err_str
                        or "429" in err_str
                        or "quota" in err_str
                        or "tokens per day" in err_str
                    )

                    if is_rate_limit:
                        # Don't waste retries on a rate-limited key — fail over now
                        print(f"   ⚠️  Rate limit hit on key #{self.current_index + 1}: {str(e)[:120]}")
                        if self._mark_exhausted_and_rotate():
                            break  # break inner retry loop, outer loop picks new client
                        else:
                            print("   ❌ All keys exhausted.")
                            raise

                    if attempt < max_retries_per_key:
                        delay = base_delay * (2 ** (attempt - 1))  # 1s, 2s, 4s
                        print(f"   ⚠️  Groq call failed (attempt {attempt}/{max_retries_per_key}, key #{self.current_index + 1}): "
                              f"{type(e).__name__}: {str(e)[:120]}")
                        print(f"      Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        print(f"   ❌ Groq call failed after {max_retries_per_key} attempts on key #{self.current_index + 1}: "
                              f"{type(e).__name__}: {e}")
                        # Exhausted retries on this key — raise, don't silently fail over
                        # (because this is NOT a rate limit; it's a real error)
                        raise
            # If we got here via `break`, we rotated — continue outer loop to use new client


def _load_groq_keys() -> list:
    """Collect all Groq API keys from the environment (primary + backup)."""
    keys = []
    primary = os.getenv("GROQ_API_KEY")
    if primary:
        keys.append(primary)
    backup = os.getenv("GROQ_API_KEY_BACKUP")
    if backup:
        keys.append(backup)
    return keys


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
# TOOL DISPATCH TABLE
# Maps tool_name → lambda(tool_input, apps) → dict result
# ─────────────────────────────────────────────
TOOL_DISPATCH = {
    # ── Email ─────────────────────────────────────────────
    "read_inbox":         lambda ti, apps: apps["email"].read_inbox(),
    "read_email":         lambda ti, apps: apps["email"].read_email(ti["email_id"]),
    "send_email":         lambda ti, apps: apps["email"].send_email(ti["to"], ti["subject"], ti["body"]),
    "reply_email":        lambda ti, apps: apps["email"].reply_email(ti["email_id"], ti["body"]),

    # ── Chat ──────────────────────────────────────────────
    "list_channels":      lambda ti, apps: apps["chat"].list_channels(),
    "read_channel":       lambda ti, apps: apps["chat"].read_channel(ti["channel"]),
    "post_message":       lambda ti, apps: apps["chat"].post_message(ti["channel"], ti["message"]),

    # ── CRM ───────────────────────────────────────────────
    "get_deal":           lambda ti, apps: apps["crm"].get_deal(ti["deal_id"]),
    "update_deal_stage":  lambda ti, apps: apps["crm"].update_deal_stage(ti["deal_id"], ti["new_stage"]),
    "add_note":           lambda ti, apps: apps["crm"].add_note(ti["deal_id"], ti["note"]),
    "get_contact":        lambda ti, apps: apps["crm"].get_contact(ti["email"]),
    "create_contact":     lambda ti, apps: apps["crm"].create_contact(
                              ti["name"], ti["email"], ti["company"], ti.get("phone", "")
                          ),

    # ── Tasks ─────────────────────────────────────────────
    "list_tasks":         lambda ti, apps: apps["tasks"].list_tasks(),
    "get_task":           lambda ti, apps: apps["tasks"].get_task(ti["task_id"]),
    "create_task":        lambda ti, apps: apps["tasks"].create_task(
                              ti["title"], ti["assigned_to"], ti.get("priority", "Medium")
                          ),
    "assign_task":        lambda ti, apps: apps["tasks"].assign_task(ti["task_id"], ti["user"]),
    "close_task":         lambda ti, apps: apps["tasks"].close_task(ti["task_id"]),

    # ── Calendar ──────────────────────────────────────────
    "list_meetings":      lambda ti, apps: apps["calendar"].list_meetings(ti.get("date")),
    "check_conflicts":    lambda ti, apps: apps["calendar"].check_conflicts(
                              ti["date"], ti["time"], ti["attendees"]
                          ),
    "book_meeting":       lambda ti, apps: apps["calendar"].book_meeting(
                              ti["title"], ti["attendees"],
                              ti["date"], ti["time"],
                              ti.get("duration_mins", 60)
                          ),
}


def execute_tool(tool_name: str, tool_input: dict, apps: dict) -> str:
    """
    Dispatches a tool call to the right app method and returns the result as JSON.
    Wraps execution in try/except so errors become tool results the agent can see.
    """
    handler = TOOL_DISPATCH.get(tool_name)
    if handler is None:
        return json.dumps({"error": f"Unknown tool: {tool_name}"}, indent=2)

    try:
        result = handler(tool_input, apps)
    except KeyError as e:
        result = {"error": f"Tool '{tool_name}' missing required argument: {e}"}
    except Exception as e:
        result = {"error": f"Tool '{tool_name}' raised an exception: {type(e).__name__}: {str(e)}"}

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

    # ── GROQ CLIENT POOL (supports primary + backup key failover) ──
    keys = _load_groq_keys()
    if not keys:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Create a .env file in the project root with:\n"
            "  GROQ_API_KEY=your_key_here\n"
            "Optionally also add a backup for automatic failover:\n"
            "  GROQ_API_KEY_BACKUP=your_second_key_here\n"
            "Get free keys at https://console.groq.com"
        )
    pool = GroqClientPool(keys)

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

        # Call Groq with automatic failover + retry
        response = pool.call(
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
        tool_calls = message.tool_calls

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

        for tc in tool_calls:
            tool_name = tc.function.name
            try:
                tool_input = json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                tool_input = {}
                print(f"   ⚠️  Failed to parse tool arguments for {tool_name}: {e}")

            tool_id = tc.id

            print(f"   🔧 Tool: {tool_name}({json.dumps(tool_input)[:80]})")

            tool_result = execute_tool(tool_name, tool_input, apps)
            print(f"   📦 Result: {tool_result[:100]}...")

            step = {
                "iteration":   iteration,
                "tool_name":   tool_name,
                "tool_input":  tool_input,
                "tool_result": tool_result,
            }
            agent_steps.append(step)
            taken_actions.append(tool_name)

            if progress_callback:
                progress_callback(step)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": tool_result,
            })
    else:
        hit_iteration_limit = True
        print(f"   ⚠️  Hit iteration limit ({max_iterations}) without agent stopping")

    # ── EPISODE COMPLETE ─────────────────────────────────────
    required_set = set(scenario["required_actions"])
    taken_set    = set(taken_actions)
    tools_ok     = required_set.issubset(taken_set)

    state_ok = True
    state_reasons = ["(no state check defined for this scenario)"]
    if "success_check" in scenario and callable(scenario["success_check"]):
        try:
            state_ok, state_reasons = scenario["success_check"](apps)
        except Exception as e:
            state_ok = False
            state_reasons = [f"❌ State check raised an exception: {str(e)}"]

    finished_cleanly = not hit_iteration_limit
    task_success = tools_ok and state_ok and finished_cleanly

    reward_result = calculate_reward(
        required_actions=scenario["required_actions"],
        taken_actions=taken_actions,
        task_success=task_success,
    )

    reward_result["state_check_passed"]  = state_ok
    reward_result["state_check_reasons"] = state_reasons
    reward_result["tools_check_passed"]  = tools_ok
    reward_result["finished_cleanly"]    = finished_cleanly

    reward_result["breakdown"].append("")
    reward_result["breakdown"].append("── State check ──")
    for line in state_reasons:
        reward_result["breakdown"].append(line)
    if not finished_cleanly:
        reward_result["breakdown"].append("⚠️  Agent hit the iteration limit without stopping cleanly")

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