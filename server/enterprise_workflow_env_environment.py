# ============================================================
# enterprise_workflow_env_environment.py
# OpenEnv-compliant environment for Enterprise Workflow tasks.
# ============================================================

import json
import os
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import EnterpriseAction, EnterpriseObservation
except ImportError:
    from models import EnterpriseAction, EnterpriseObservation

try:
    from .mock_apps import EmailApp, ChatApp, CRMApp, TaskApp, CalendarApp
    from .scenarios import SCENARIOS, get_scenario_instance
    from .reward import calculate_reward
except ImportError:
    from mock_apps import EmailApp, ChatApp, CRMApp, TaskApp, CalendarApp
    from scenarios import SCENARIOS, get_scenario_instance
    from reward import calculate_reward

DEFAULT_SCENARIO_ID = os.getenv("DEFAULT_SCENARIO_ID", "scenario_deal_rescue")
MAX_STEPS_PER_EPISODE = int(os.getenv("MAX_STEPS_PER_EPISODE", "40"))

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


def _dispatch_tool(tool_name, tool_args, apps):
    handler = TOOL_DISPATCH.get(tool_name)
    if handler is None:
        return None, "Unknown tool: " + tool_name
    try:
        return handler(tool_args, apps), ""
    except KeyError as e:
        return None, "Tool " + tool_name + " missing required argument: " + str(e)
    except Exception as e:
        return None, "Tool " + tool_name + " raised " + type(e).__name__ + ": " + str(e)


class EnterpriseWorkflowEnvironment(Environment):
    """OpenEnv environment for long-horizon, multi-app enterprise workflows."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._scenario = get_scenario_instance(DEFAULT_SCENARIO_ID)
        self._apps = {}
        self._taken_actions = []
        self._done = False
        self._episode_score = 0
        self._create_fresh_apps()

    def _create_fresh_apps(self):
        self._apps = {
            "email":    EmailApp(),
            "chat":     ChatApp(),
            "crm":      CRMApp(),
            "tasks":    TaskApp(),
            "calendar": CalendarApp(),
        }

    def _make_observation(self, tool_result="", error="", reward=0.0,
                          breakdown=None, task_success=None):
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

    def _check_terminal(self):
        if self._state.step_count >= MAX_STEPS_PER_EPISODE:
            return True
        required = set(self._scenario["required_actions"])
        if not required.issubset(set(self._taken_actions)):
            return False
        success_check = self._scenario.get("success_check")
        if success_check is None:
            return True
        try:
            state_ok, _ = success_check(self._apps)
        except Exception:
            return False
        return bool(state_ok)

    def _compute_terminal_reward(self):
        required_actions = list(self._scenario["required_actions"])
        required_set = set(required_actions)
        taken_set = set(self._taken_actions)
        tools_ok = required_set.issubset(taken_set)

        state_ok = True
        state_reasons = ["(no state check defined)"]
        success_check = self._scenario.get("success_check")
        if success_check is not None:
            try:
                state_ok, state_reasons = success_check(self._apps)
            except Exception as e:
                state_ok = False
                state_reasons = ["State check raised: " + str(e)]

        finished_cleanly = self._state.step_count < MAX_STEPS_PER_EPISODE
        task_success = tools_ok and state_ok and finished_cleanly

        reward_result = calculate_reward(
            required_actions=required_actions,
            taken_actions=list(self._taken_actions),
            task_success=task_success,
            expected_counts=self._scenario.get("expected_counts", {}),
        )

        breakdown = list(reward_result["breakdown"])
        breakdown.append("")
        breakdown.append("State check:")
        for line in state_reasons:
            breakdown.append(line)
        if not finished_cleanly:
            breakdown.append("Step cap reached")

        return reward_result["score"], breakdown, task_success

    def reset(self):
        scenario_id = os.getenv("DEFAULT_SCENARIO_ID", DEFAULT_SCENARIO_ID)
        self._scenario = get_scenario_instance(scenario_id)
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

    def step(self, action):
        if self._done:
            return self._make_observation(
                tool_result=json.dumps({"status": "episode_already_done"}),
                reward=0.0,
            )

        self._state.step_count += 1
        result, error = _dispatch_tool(action.tool_name, action.tool_args, self._apps)

        if error and error.startswith("Unknown tool"):
            return self._make_observation(error=error, reward=0.0)

        self._taken_actions.append(action.tool_name)
        tool_result_str = json.dumps(result, indent=2) if result is not None else ""

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

        return self._make_observation(
            tool_result=tool_result_str,
            error=error,
            reward=0.0,
        )

    @property
    def state(self):
        return self._state
