# ============================================================
# enterprise_workflow_env_environment.py
#
# The OpenEnv-compliant environment for the Enterprise Workflow task.
# Wraps our existing mock apps, reward function, and scenarios so they
# can be served over HTTP/WebSocket and trained against via TRL.
#
# ── Episode lifecycle ────────────────────────────────────────
#   1. reset() is called. The env loads a scenario (by ID or default),
#      creates fresh instances of all 5 mock apps, and returns the
#      initial observation (empty tool_result, scenario context).
#
#   2. step(action) is called once per tool the agent wants to invoke.
#      The tool dispatches to the correct mock app method. The
#      per-step reward is 0 (sparse reward — matches hackathon
#      Theme #2 "long-horizon with sparse/delayed rewards").
#
#   3. The episode terminates when:
#        (a) the agent has called all required tools AND the state
#            check passes, OR
#        (b) a safety cap on step count is reached.
#      On termination, the full reward (from calculate_reward) is
#      returned in the terminal observation.
#
# ── Scenario selection ──────────────────────────────────────
# The scenario is selected at reset() time via an environment variable
# (DEFAULT_SCENARIO_ID), or via a Scenario-Id HTTP header that the
# training client can set per-episode. Defaults to "scenario_deal_rescue".
# ============================================================

import json
import os
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# Import our local models and reused business logic
try:
    from ..models import EnterpriseAction, EnterpriseObservation
except ImportError:
    from models import EnterpriseAction, EnterpriseObservation

# These three files were copied from the project root into server/
# These three files were copied from the project root into server/.
# Try relative import first (uvicorn/package mode), fall back to absolute
# (direct-script mode, which is how we've been testing).
try:
    from .mock_apps import EmailApp, ChatApp, CRMApp, TaskApp, CalendarApp
    from .scenarios import SCENARIOS
    from .reward import calculate_reward
except ImportError:
    from mock_apps import EmailApp, ChatApp, CRMApp, TaskApp, CalendarApp
    from scenarios import SCENARIOS
    from reward import calculate_reward


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DEFAULT_SCENARIO_ID = os.getenv("DEFAULT_SCENARIO_ID", "scenario_deal_rescue")

# Safety cap so a runaway agent can't loop forever during training
MAX_STEPS_PER_EPISODE = int(os.getenv("MAX_STEPS_PER_EPISODE", "40"))


# ─────────────────────────────────────────────
# TOOL DISPATCH (ported from agent.py)
# ─────────────────────────────────────────────
# Each lambda takes the tool arguments dict + the apps dict and returns
# whatever the underlying mock app method returns (a dict).
# This is the same table we use in our Streamlit agent loop — single
# source of truth.
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


def _dispatch_tool(tool_name: str, tool_args: dict, apps: dict):
    """
    Run one tool call. Returns (result_dict, error_string).
    Exactly one of the two is populated.
    """
    handler = TOOL_DISPATCH.get(tool_name)
    if handler is None:
        return None, f"Unknown tool: {tool_name}"

    try:
        return handler(tool_args, apps), ""
    except KeyError as e:
        return None, f"Tool '{tool_name}' missing required argument: {e}"
    except Exception as e:
        return None, f"Tool '{tool_name}' raised {type(e).__name__}: {e}"


def _find_scenario(scenario_id: str) -> dict:
    """Look up a scenario dict by ID. Falls back to the default."""
    for s in SCENARIOS:
        if s["id"] == scenario_id:
            return s
    # Fall back to the first scenario if the ID is unrecognized
    return SCENARIOS[0]


# ─────────────────────────────────────────────
# THE ENVIRONMENT
# ─────────────────────────────────────────────
class EnterpriseWorkflowEnvironment(Environment):
    """
    OpenEnv environment for long-horizon, multi-app enterprise workflows.

    Each episode runs ONE scenario (Deal Rescue, Team Conflict, or
    Client Onboarding). The agent takes actions (tool calls) across
    5 mock apps (Email, Chat, CRM, Tasks, Calendar). The episode
    ends when the agent either succeeds (all required tools called
    AND state check passes) or exceeds the step cap.

    Reward is sparse: 0 on every intermediate step, and the full
    episode score on the terminal step.
    """

    # Concurrent WebSocket sessions — required for TRL GRPO training which
    # opens N connections (one per generation) against the same server.
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize with default scenario. Fresh apps created on first reset."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._scenario = _find_scenario(DEFAULT_SCENARIO_ID)
        self._apps: dict = {}
        self._taken_actions: list = []
        self._done: bool = False
        self._episode_score: int = 0
        # Ensure apps exist even before the first reset() — defensive.
        self._create_fresh_apps()

    # ── Helpers ─────────────────────────────────────────────
    def _create_fresh_apps(self) -> None:
        """Fresh instances of all 5 apps (isolated per episode)."""
        self._apps = {
            "email":    EmailApp(),
            "chat":     ChatApp(),
            "crm":      CRMApp(),
            "tasks":    TaskApp(),
            "calendar": CalendarApp(),
        }

    def _make_observation(
        self,
        tool_result: str = "",
        error: str = "",
        reward: float = 0.0,
        breakdown=None,
        task_success=None,
    ) -> EnterpriseObservation:
        """Build an observation with the current episode context attached."""
        return EnterpriseObservation(
            tool_result=tool_result,
            error=error,
            scenario_id=self._scenario["id"],
            taken_actions=list(self._taken_actions),
            step_count=self._state.step_count,
            breakdown=breakdown or [],
            task_success=task_success,
            done=self._done,
            reward=reward,
        )

    def _check_terminal(self) -> bool:
        """
        Decide whether the episode is over.

        Terminal conditions:
          - Required tools all called AND scenario-specific state check passes, OR
          - Step cap exceeded (runaway safety net)
        """
        if self._state.step_count >= MAX_STEPS_PER_EPISODE:
            return True

        required = set(self._scenario["required_actions"])
        if not required.issubset(set(self._taken_actions)):
            return False

        # All required tools have been called at least once.
        # Now check the scenario-specific state-based success condition.
        success_check = self._scenario.get("success_check")
        if success_check is None:
            return True
        try:
            state_ok, _ = success_check(self._apps)
        except Exception:
            # If the state check crashes, treat as non-terminal — the
            # agent can keep trying until the step cap saves us.
            return False
        return bool(state_ok)

    def _compute_terminal_reward(self):
        """
        Run the full reward function on the completed episode.
        Returns (score, breakdown, task_success).
        """
        required_actions = list(self._scenario["required_actions"])
        required_set = set(required_actions)
        taken_set = set(self._taken_actions)
        tools_ok = required_set.issubset(taken_set)

        # State check
        state_ok = True
        state_reasons = ["(no state check defined for this scenario)"]
        success_check = self._scenario.get("success_check")
        if success_check is not None:
            try:
                state_ok, state_reasons = success_check(self._apps)
            except Exception as e:
                state_ok = False
                state_reasons = [f"❌ State check raised: {e}"]

        finished_cleanly = self._state.step_count < MAX_STEPS_PER_EPISODE
        task_success = tools_ok and state_ok and finished_cleanly

        reward_result = calculate_reward(
            required_actions=required_actions,
            taken_actions=list(self._taken_actions),
            task_success=task_success,
            expected_counts=self._scenario.get("expected_counts", {}),
        )

        # Append the state-check lines to the breakdown so the observation
        # carries the full explanation of the score (matches Streamlit UI).
        breakdown = list(reward_result["breakdown"])
        breakdown.append("")
        breakdown.append("── State check ──")
        for line in state_reasons:
            breakdown.append(line)
        if not finished_cleanly:
            breakdown.append("⚠️  Step cap reached without agent completing cleanly")

        return reward_result["score"], breakdown, task_success

    # ── OpenEnv API — reset ─────────────────────────────────
    def reset(self) -> EnterpriseObservation:
        """Start a new episode. Optionally picks a scenario via DEFAULT_SCENARIO_ID."""
        # Re-read the env var in case the training script sets it per-episode
        scenario_id = os.getenv("DEFAULT_SCENARIO_ID", DEFAULT_SCENARIO_ID)
        self._scenario = _find_scenario(scenario_id)

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._taken_actions = []
        self._done = False
        self._episode_score = 0
        self._create_fresh_apps()

        return self._make_observation(
            tool_result=json.dumps({
                "status": "ready",
                "scenario": self._scenario["name"],
                "description": self._scenario["description"],
                "agent_prompt": self._scenario["agent_prompt"],
                "required_actions": self._scenario["required_actions"],
            }),
            reward=0.0,
        )

    # ── OpenEnv API — step ──────────────────────────────────
    def step(self, action: EnterpriseAction) -> EnterpriseObservation:
        """Execute one tool call and advance the episode."""
        if self._done:
            # Episode already over — return a terminal observation with zero reward
            return self._make_observation(
                tool_result=json.dumps({"status": "episode_already_done"}),
                reward=0.0,
            )

        self._state.step_count += 1

        # Dispatch the tool
        result, error = _dispatch_tool(action.tool_name, action.tool_args, self._apps)

        # Track what the agent did (only for known tools — unknowns are pure noise)
        if error and error.startswith("Unknown tool"):
            # Don't record the bad name in taken_actions — it shouldn't count
            # toward progress. But we still return an error observation.
            return self._make_observation(error=error, reward=0.0)

        self._taken_actions.append(action.tool_name)

        tool_result_str = json.dumps(result, indent=2) if result is not None else ""

        # Check terminality
        if self._check_terminal():
            score, breakdown, task_success = self._compute_terminal_reward()
            self._done = True
            self._episode_score = score
            return self._make_observation(
                tool_result=tool_result_str,
                error=error,
                reward=float(score),
                breakdown=breakdown,
                task_success=task_success,
            )

        # Non-terminal: return tool result with 0 reward (sparse reward)
        return self._make_observation(
            tool_result=tool_result_str,
            error=error,
            reward=0.0,
        )

    # ── OpenEnv API — state ─────────────────────────────────
    @property
    def state(self) -> State:
        """Current episode state (used by the /state HTTP endpoint)."""
        return self._state