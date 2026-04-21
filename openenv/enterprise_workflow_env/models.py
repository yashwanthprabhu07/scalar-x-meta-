# ============================================================
# models.py — OpenEnv data models for the Enterprise Workflow Env
#
# These Pydantic models define the wire format for the environment:
#   - EnterpriseAction: what the agent sends to the env (a tool call)
#   - EnterpriseObservation: what the env returns (tool result + reward)
#   - State fields captured in the State object (episode_id, step_count,
#     scenario_id, done, cumulative reward)
#
# The Action uses a discriminated-union style: `tool_name` picks which
# of the 20 tools to invoke, and `tool_args` carries the arguments.
# This is the OpenEnv-compliant shape.
#
# (The TRL training script, in a later step, will wrap this environment
# with individually-named tool methods since TRL's GRPOTrainer prefers
# typed tool-calling over a single generic step().)
# ============================================================

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


# ─────────────────────────────────────────────
# ACTION
# ─────────────────────────────────────────────
class EnterpriseAction(Action):
    """
    Action sent by the agent to the enterprise workflow environment.

    The agent picks a tool from the 20 available and supplies its arguments.

    Examples:
        EnterpriseAction(tool_name="read_inbox", tool_args={})
        EnterpriseAction(tool_name="assign_task", tool_args={"task_id": "TASK-003", "user": "Sneha"})
        EnterpriseAction(tool_name="book_meeting", tool_args={
            "title": "Sync",
            "attendees": ["arjun", "sneha"],
            "date": "2026-04-22",
            "time": "11:00",
            "duration_mins": 15,
        })
    """

    tool_name: str = Field(
        ...,
        description=(
            "Name of the tool to invoke. Must be one of the 20 registered tools: "
            "read_inbox, read_email, send_email, reply_email, "
            "list_channels, read_channel, post_message, "
            "get_deal, update_deal_stage, add_note, get_contact, create_contact, "
            "list_tasks, get_task, create_task, assign_task, close_task, "
            "list_meetings, check_conflicts, book_meeting."
        ),
    )
    tool_args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for the tool call (tool-specific keys and values).",
    )


# ─────────────────────────────────────────────
# OBSERVATION
# ─────────────────────────────────────────────
class EnterpriseObservation(Observation):
    """
    Observation returned by the environment after each step.

    Semantic notes:
      - `tool_result` is the JSON-serialized result the tool produced
        (matches what the mock app method returned).
      - `error` is non-empty only if the tool raised or the tool_name was unknown.
      - `done` is True only when the agent has called every required tool AND
        the scenario's state-based success check passes (or when the episode
        hits a safety limit — see the Environment implementation).
      - `reward` is the per-step reward. For long-horizon tasks, most steps
        return 0; only the terminal step returns the full episode score
        (this is the standard sparse-reward pattern that matches Theme #2
        of the hackathon).
      - `scenario_id` echoes the active scenario so clients know what's running.
      - `taken_actions` is the running list of tool names called so far —
        useful for clients that want to inspect trajectory during rollout.
      - `step_count` is how many steps have been taken in this episode.
      - `breakdown` is populated on the terminal step with the full reward
        breakdown (the ✅/❌ bullets from our existing reward function).
    """

    tool_result: str = Field(default="", description="JSON-serialized tool result")
    error: str = Field(default="", description="Error message if the tool failed; empty on success")
    scenario_id: str = Field(default="", description="ID of the active scenario")
    taken_actions: List[str] = Field(
        default_factory=list,
        description="List of tool names called in this episode so far",
    )
    step_count: int = Field(default=0, description="Number of steps taken in this episode")
    breakdown: List[str] = Field(
        default_factory=list,
        description="Reward breakdown (populated only on the terminal step)",
    )
    task_success: Optional[bool] = Field(
        default=None,
        description="Whether the scenario was completed successfully (terminal step only)",
    )