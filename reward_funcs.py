# ============================================================
# reward_funcs.py — Independent reward functions for GRPO training
#
# The hackathon build guide explicitly recommends multiple independent
# reward functions instead of a single scalar (Section 7: "If you only
# have a single reward signal, it is easier for the model to hack it").
#
# TRL's GRPOTrainer takes `reward_funcs=[fn1, fn2, ...]` — it sums them
# per prompt, but also tracks each one separately in metrics. This lets
# us see in training plots exactly which axis the agent is learning on.
#
# Each function here takes a `trajectory` dict (as returned by
# run_episode) and returns a float. Compose them into a list for TRL.
#
# Axes:
#   1. tool_correctness  — did the agent call the required tools?
#   2. tool_efficiency   — did it avoid wasteful duplicate calls?
#   3. task_completion   — did the final state match what we wanted?
#   4. format_validity   — were tool calls well-formed (no unknowns)?
# ============================================================

from collections import Counter
from typing import Any, Dict, List


# ─────────────────────────────────────────────
# 1. TOOL CORRECTNESS
# ─────────────────────────────────────────────
def tool_correctness_reward(trajectory: Dict[str, Any]) -> float:
    """
    +1 for each required tool the agent called at least once.

    Independent of duplicate calls — we only care here that the
    agent recognized each required tool. Efficiency is a separate
    function below.

    Max reward = N_required.
    """
    required = set(trajectory.get("required_actions", []))
    taken    = set(trajectory.get("taken_actions", []))
    return float(len(required & taken))


# ─────────────────────────────────────────────
# 2. TOOL EFFICIENCY
# ─────────────────────────────────────────────
def tool_efficiency_reward(trajectory: Dict[str, Any]) -> float:
    """
    Penalizes duplicate and unnecessary tool calls.

    -0.5 per duplicate (call beyond expected count)
    -1.0 per unnecessary call (tool not in required_actions)

    Scenarios can declare expected_counts in their definition to
    allow legitimate multi-calls (e.g. assign_task: 2 for Team
    Conflict). Default expected count for a required tool is 1.

    Max reward = 0 (no penalty for perfect play).
    """
    required = set(trajectory.get("required_actions", []))
    taken    = trajectory.get("taken_actions", [])
    expected_counts = trajectory.get("expected_counts", {})

    taken_counter = Counter(taken)
    penalty = 0.0

    for tool, count in taken_counter.items():
        if tool not in required:
            # Tool wasn't needed at all
            penalty += count * 1.0
        else:
            expected = expected_counts.get(tool, 1)
            if count > expected:
                penalty += (count - expected) * 0.5

    return -penalty


# ─────────────────────────────────────────────
# 3. TASK COMPLETION
# ─────────────────────────────────────────────
def task_completion_reward(trajectory: Dict[str, Any]) -> float:
    """
    The big payoff — +10 if the scenario's state-based success check
    passed, 0 otherwise.

    This is a sparse-but-strong signal that prevents the agent from
    just rapid-firing tools. The whole reward stack doesn't pay off
    without genuine completion.
    """
    return 10.0 if trajectory.get("task_success") else 0.0


# ─────────────────────────────────────────────
# 4. FORMAT VALIDITY
# ─────────────────────────────────────────────
def format_validity_reward(trajectory: Dict[str, Any]) -> float:
    """
    Rewards well-formed tool calls:
    +2 if the agent produced only valid tool names with usable args
    0  if any step had an error (unknown tool, missing required arg, etc.)

    This catches the common LLM failure mode of hallucinating tools
    or forgetting args.
    """
    steps = trajectory.get("steps", [])
    if not steps:
        return 0.0

    any_error = any(step.get("error") for step in steps)
    return 0.0 if any_error else 2.0


# ─────────────────────────────────────────────
# ALL REWARD FUNCTIONS (for convenience)
# ─────────────────────────────────────────────
ALL_REWARD_FNS = [
    ("tool_correctness", tool_correctness_reward),
    ("tool_efficiency",  tool_efficiency_reward),
    ("task_completion",  task_completion_reward),
    ("format_validity",  format_validity_reward),
]


def score_trajectory(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate all reward functions on a trajectory. Returns a dict:
      {
        "tool_correctness": 5.0,
        "tool_efficiency":  -1.0,
        "task_completion":  10.0,
        "format_validity":  2.0,
        "total":            16.0,
        "breakdown":        [ "✅ tool_correctness: 5.0", ... ],
      }
    """
    scores = {}
    breakdown = []
    total = 0.0
    for name, fn in ALL_REWARD_FNS:
        v = float(fn(trajectory))
        scores[name] = v
        total += v
        sign = "✅" if v >= 0 else "⚠️"
        breakdown.append(f"{sign} {name}: {v:+.1f}")
    scores["total"] = total
    scores["breakdown"] = breakdown
    return scores


# ─────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────
def _smoke_test():
    """Test each reward function on realistic trajectory shapes."""
    perfect = {
        "required_actions": ["read_inbox", "read_email", "reply_email"],
        "taken_actions":    ["read_inbox", "read_email", "reply_email"],
        "task_success":     True,
        "steps":            [{"error": ""}, {"error": ""}, {"error": ""}],
        "expected_counts":  {},
    }
    duplicated = {
        **perfect,
        "taken_actions": ["read_inbox", "read_inbox", "read_email", "reply_email"],
    }
    unnecessary = {
        **perfect,
        "taken_actions": ["read_inbox", "list_channels", "read_email", "reply_email"],
    }
    errored = {
        **perfect,
        "steps": [{"error": ""}, {"error": "Unknown tool: wut"}, {"error": ""}],
    }
    failed = {
        **perfect,
        "task_success": False,
    }

    print("── Perfect trajectory ──")
    print(score_trajectory(perfect))
    print("\n── Trajectory with 1 duplicate ──")
    print(score_trajectory(duplicated))
    print("\n── Trajectory with 1 unnecessary tool ──")
    print(score_trajectory(unnecessary))
    print("\n── Trajectory with 1 errored step ──")
    print(score_trajectory(errored))
    print("\n── Trajectory that didn't complete ──")
    print(score_trajectory(failed))


if __name__ == "__main__":
    _smoke_test()