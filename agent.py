# ============================================================
# agent.py - The AI Agent Loop (Using Groq API - free & fast)
# ============================================================
import os
import sys
import json
import time
from groq import Groq
from dotenv import load_dotenv
 
load_dotenv()
 
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if callable(_reconfigure):
        try:
            _reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
 
from mock_apps import EmailApp, ChatApp, CRMApp, TaskApp, CalendarApp
from tools import ALL_TOOLS
from reward import calculate_reward, log_episode
 
from memory import format_lessons_for_prompt, add_lesson, get_lessons
from lesson_extractor import extract_lesson
 
 
# ---------------------------------------------
# GROQ CLIENT POOL
# ---------------------------------------------
class GroqClientPool:
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
        self.exhausted = [False] * len(api_keys)
        print(f"   [KEY] Groq client pool initialized with {len(api_keys)} key(s)")
 
    def current_client(self) -> Groq:
        return self.clients[self.current_index]
 
    def _mark_exhausted_and_rotate(self):
        self.exhausted[self.current_index] = True
        print(f"   [ROTATE] Key #{self.current_index + 1} exhausted, trying next key...")
        for i in range(len(self.clients)):
            next_idx = (self.current_index + 1 + i) % len(self.clients)
            if not self.exhausted[next_idx]:
                self.current_index = next_idx
                print(f"   [OK] Failed over to key #{self.current_index + 1}")
                return True
        return False
 
    def call(self, *, model, max_tokens, tools, tool_choice, messages,
             max_retries_per_key: int = 3, base_delay: float = 1.0,
             request_timeout: float = 90.0):
        last_exception = None
 
        while True:
            client = self.current_client()
 
            for attempt in range(1, max_retries_per_key + 1):
                try:
                    return client.with_options(timeout=request_timeout).chat.completions.create(
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
                        print(f"   [WARN]  Rate limit hit on key #{self.current_index + 1}: {str(e)[:120]}")
                        if self._mark_exhausted_and_rotate():
                            break
                        else:
                            print("   [FAIL] All keys exhausted.")
                            raise
 
                    if attempt < max_retries_per_key:
                        delay = base_delay * (2 ** (attempt - 1))
                        print(f"   [WARN]  Groq call failed (attempt {attempt}/{max_retries_per_key}, key #{self.current_index + 1}): "
                              f"{type(e).__name__}: {str(e)[:120]}")
                        print(f"      Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        print(f"   [FAIL] Groq call failed after {max_retries_per_key} attempts on key #{self.current_index + 1}: "
                              f"{type(e).__name__}: {e}")
                        raise
 
 
def _load_groq_keys() -> list:
    keys = []
    primary = os.getenv("GROQ_API_KEY")
    if primary:
        keys.append(primary)
    backup = os.getenv("GROQ_API_KEY_BACKUP")
    if backup:
        keys.append(backup)
    return keys
 
 
# ---------------------------------------------
# FRESH APPS PER EPISODE
# ---------------------------------------------
def create_fresh_apps():
    return {
        "email":    EmailApp(),
        "chat":     ChatApp(),
        "crm":      CRMApp(),
        "tasks":    TaskApp(),
        "calendar": CalendarApp(),
    }
 
 
# ---------------------------------------------
# TOOL DISPATCH TABLE
# ---------------------------------------------
TOOL_DISPATCH = {
    "read_inbox":         lambda ti, apps: apps["email"].read_inbox(),
    "read_email":         lambda ti, apps: apps["email"].read_email(ti["email_id"]),
    "send_email":         lambda ti, apps: apps["email"].send_email(ti["to"], ti["subject"], ti["body"]),
    "reply_email":        lambda ti, apps: apps["email"].reply_email(ti["email_id"], ti["body"]),
 
    "list_channels":      lambda ti, apps: apps["chat"].list_channels(),
    "read_channel":       lambda ti, apps: apps["chat"].read_channel(ti["channel"]),
    "post_message":       lambda ti, apps: apps["chat"].post_message(ti["channel"], ti["message"]),
 
    "get_deal":           lambda ti, apps: apps["crm"].get_deal(ti["deal_id"]),
    "update_deal_stage":  lambda ti, apps: apps["crm"].update_deal_stage(ti["deal_id"], ti["new_stage"]),
    "add_note":           lambda ti, apps: apps["crm"].add_note(ti["deal_id"], ti["note"]),
    "get_contact":        lambda ti, apps: apps["crm"].get_contact(ti["email"]),
    "create_contact":     lambda ti, apps: apps["crm"].create_contact(
                              ti["name"], ti["email"], ti["company"], ti.get("phone", "")
                          ),
 
    "list_tasks":         lambda ti, apps: apps["tasks"].list_tasks(),
    "get_task":           lambda ti, apps: apps["tasks"].get_task(ti["task_id"]),
    "create_task":        lambda ti, apps: apps["tasks"].create_task(
                              ti["title"], ti["assigned_to"], ti.get("priority", "Medium")
                          ),
    "assign_task":        lambda ti, apps: apps["tasks"].assign_task(ti["task_id"], ti["user"]),
    "close_task":         lambda ti, apps: apps["tasks"].close_task(ti["task_id"]),
 
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
 
 
# ---------------------------------------------
# BASE SYSTEM PROMPT
# ---------------------------------------------
_BASE_SYSTEM_PROMPT = (
    "You are an efficient AI employee at a company. "
    "You have access to 5 company apps: Email, Chat, CRM, Tasks, and Calendar. "
    "Complete all the steps in the user's task using the available tools. "
    "Be thorough - use every tool needed to fully complete the task. "
    "IMPORTANT: Do not call the same tool more than once unless you have a clear new reason. "
    "Before acting, read the relevant state first (inbox, channel, task). "
    "If a tool returns an error, examine the error message and try a different approach."
)
 
 
# ---------------------------------------------
# MAIN AGENT LOOP
# ---------------------------------------------
def run_agent(scenario: dict, progress_callback=None):
    apps = create_fresh_apps()
 
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
 
    scenario_id   = scenario["id"]
    lessons_text  = format_lessons_for_prompt(scenario_id)
    prior_lessons = get_lessons(scenario_id)
 
    if prior_lessons:
        print(f"   [MEM] Loaded {len(prior_lessons)} lesson(s) from memory for this scenario")
        for i, lesson in enumerate(prior_lessons, 1):
            print(f"      {i}. {lesson['text'][:100]}")
    else:
        print(f"   [MEM] No prior lessons for this scenario - running without memory")
 
    system_prompt = _BASE_SYSTEM_PROMPT + lessons_text
 
    agent_steps   = []
    taken_actions = []
 
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
 
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": scenario["agent_prompt"]},
    ]
 
    print(f"\n{'='*60}")
    print(f"RUNNING SCENARIO: {scenario['name']}")
    print(f"{'='*60}\n")
 
    max_iterations    = 20
    hit_iteration_limit = False
 
    for iteration in range(1, max_iterations + 1):
        print(f"[Iteration {iteration}] Asking Groq what to do next...")
 
        response = pool.call(
            model="llama-3.3-70b-versatile",
            max_tokens=4096,
            tools=grok_tools,
            tool_choice="auto",
            messages=messages,
        )
 
        message       = response.choices[0].message
        finish_reason = response.choices[0].finish_reason
        print(f"   -> Finish reason: {finish_reason}")
 
        if finish_reason == "stop" or not message.tool_calls:
            final_text = message.content or "Task completed."
            print(f"   [OK] Agent finished: {final_text[:100]}")
            break
 
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
                print(f"   [WARN]  Failed to parse tool arguments for {tool_name}: {e}")
 
            tool_id = tc.id
            print(f"   [TOOL] Tool: {tool_name}({json.dumps(tool_input)[:80]})")
 
            tool_result = execute_tool(tool_name, tool_input, apps)
            print(f"   [RESULT] Result: {tool_result[:100]}...")
 
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
        print(f"   [WARN]  Hit iteration limit ({max_iterations}) without agent stopping")
 
    # -- SCORING --------------------------------------------------
    required_set = set(scenario["required_actions"])
    taken_set    = set(taken_actions)
    tools_ok     = required_set.issubset(taken_set)
 
    state_ok      = True
    state_reasons = ["(no state check defined for this scenario)"]
    if "success_check" in scenario and callable(scenario["success_check"]):
        try:
            state_ok, state_reasons = scenario["success_check"](apps)
        except Exception as e:
            state_ok      = False
            state_reasons = [f"[FAIL] State check raised an exception: {str(e)}"]
 
    finished_cleanly = not hit_iteration_limit
    task_success     = tools_ok and state_ok and finished_cleanly
 
    reward_result = calculate_reward(
        required_actions=scenario["required_actions"],
        taken_actions=taken_actions,
        task_success=task_success,
        expected_counts=scenario.get("expected_counts", {}),
    )
 
    reward_result["state_check_passed"]  = state_ok
    reward_result["state_check_reasons"] = state_reasons
    reward_result["tools_check_passed"]  = tools_ok
    reward_result["finished_cleanly"]    = finished_cleanly
 
    reward_result["breakdown"].append("")
    reward_result["breakdown"].append("-- State check --")
    for line in state_reasons:
        reward_result["breakdown"].append(line)
    if not finished_cleanly:
        reward_result["breakdown"].append("[WARN]  Agent hit the iteration limit without stopping cleanly")
 
    episode = log_episode(
        scenario_id=scenario["id"],
        reward_result=reward_result,
        agent_steps=agent_steps,
    )
 
    # ── FIX: guarantee the keys dashboard._drain_queue expects ──────────
    # log_episode may or may not write score/task_success onto the episode
    # dict directly. We inject them here from reward_result so the dashboard
    # always finds them, regardless of how reward.py is implemented.
    episode.setdefault("score",        reward_result.get("score", 0))
    episode.setdefault("task_success", reward_result.get("task_success", task_success))
    episode.setdefault("steps_taken",  len(agent_steps))
    episode.setdefault("scenario_id",  scenario["id"])
    episode.setdefault("timestamp",    time.strftime("%Y-%m-%dT%H:%M:%S"))
    # ── END FIX ──────────────────────────────────────────────────────────
 
    print(f"\n   [OK] State check complete (passed={state_ok}). Episode logged.", flush=True)
 
    # -- EXTRACT LESSON -------------------------------------------
    print(f"   [LESSON] Extracting lesson from this episode (Groq, 30s timeout)...", flush=True)
    new_lesson = extract_lesson(
        scenario_name=scenario["name"],
        agent_prompt=scenario["agent_prompt"],
        taken_actions=taken_actions,
        agent_steps=agent_steps,
        reward_result=reward_result,
        groq_client=pool.current_client(),
    )
    if new_lesson:
        add_lesson(
            scenario_id=scenario_id,
            lesson_text=new_lesson,
            episode_number=episode["episode_number"],
            score=reward_result["score"],
        )
        print(f"   [LESSON] New lesson saved: {new_lesson[:150]}", flush=True)
        reward_result["new_lesson"] = new_lesson
    else:
        print(f"   [WARN]  No lesson extracted (skipped or extractor returned empty)", flush=True)
        reward_result["new_lesson"] = ""
 
    # -- FINAL LOG ------------------------------------------------
    print(f"\n{'-'*40}", flush=True)
    print(f"SCORE: {reward_result['score']}  (normalized {reward_result['normalized_score']})", flush=True)
    print(f"SUCCESS: {task_success}  (tools_ok={tools_ok}, state_ok={state_ok}, finished_cleanly={finished_cleanly})", flush=True)
    for line in reward_result["breakdown"]:
        print(f"   {line}", flush=True)
 
    return {
        "episode": episode,
        "reward":  reward_result,
        "steps":   agent_steps,
        "apps":    apps,
    }