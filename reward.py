# ============================================================
# reward.py - Reward Function and Episode Logger
# This scores how well the agent did after each task.
# Higher score = agent did better.
# ============================================================
from collections import Counter
from datetime import datetime

# In-memory episode history (list of all past episode results)
episode_history = []


def calculate_reward(required_actions: list,
                     taken_actions: list,
                     task_success: bool,
                     expected_counts: dict = None) -> dict:
    """
    Scores the agent's performance for one episode.

    Scoring rules:
      +10 if task fully completed, else -10
      +2  for each DISTINCT required action that was correctly taken
      -5  for each required action that was MISSED
      -1  for every EXTRA call - this includes:
            * calls to tools not in required_actions
            * calls to a required tool BEYOND its expected count
              (e.g. if expected_counts says create_task: 2, then
              calling create_task 3 times costs -1; 2 times is free)

    Args:
        required_actions: list of tool names the agent MUST call
        taken_actions:    list of tool names the agent actually called (in order)
        task_success:     True if all required actions were taken
        expected_counts:  optional dict of {tool_name: expected_count}.
                          If a required tool isn't listed here, the expected
                          count defaults to 1. This lets scenarios legitimately
                          require multiple calls to the same tool (e.g. creating
                          two different tasks) without being penalized.
    """
    if expected_counts is None:
        expected_counts = {}

    score = 0
    breakdown = []

    required_set = set(required_actions)
    taken_counts = Counter(taken_actions)

    # +10 / -10 for task outcome
    if task_success:
        score += 10
        breakdown.append("+10 Task fully completed")
    else:
        score -= 10
        breakdown.append("-10 Task failed")

    # +2 for each DISTINCT required action correctly taken
    correct_actions = required_set & set(taken_counts.keys())
    for action in sorted(correct_actions):
        score += 2
        breakdown.append(f"+2 Correct tool used: {action}")

    # -5 for each required action that was MISSED
    missed_actions = required_set - set(taken_counts.keys())
    for action in sorted(missed_actions):
        score -= 5
        breakdown.append(f"-5 Required action missed: {action}")

    # -1 for each EXTRA call:
    #   * required tool called beyond its expected count
    #   * any call to a non-required tool
    extra_count = 0
    for tool, count in taken_counts.items():
        if tool in required_set:
            # Default expected count is 1; scenarios can override
            expected = expected_counts.get(tool, 1)
            extras = count - expected
            if extras > 0:
                extra_count += extras
                score -= extras
                breakdown.append(
                    f"-{extras} Excess call(s) to required tool: {tool} "
                    f"(called {count}x, expected {expected})"
                )
        else:
            # All calls to non-required tools are extras
            extra_count += count
            score -= count
            breakdown.append(
                f"-{count} Unnecessary tool call(s): {tool} (called {count}x)"
            )

    # Max possible score: +10 for success, +2 per distinct required action.
    # (We don't reward multi-count in the max - the +2 is per distinct tool.)
    max_possible = 10 + (2 * len(required_set))

    return {
        "score": score,
        "max_possible_score": max_possible,
        "normalized_score": round(score / max_possible, 3) if max_possible > 0 else 0.0,
        "task_success": task_success,
        "breakdown": breakdown,
        "required_actions": list(required_actions),
        "taken_actions": taken_actions,
        "correct_count": len(correct_actions),
        "missed_count": len(missed_actions),
        "extra_count": extra_count,
    }


def log_episode(scenario_id: str, reward_result: dict, agent_steps: list) -> dict:
    """Saves the result of one episode to history."""
    episode = {
        "episode_number": len(episode_history) + 1,
        "scenario_id": scenario_id,
        "score": reward_result["score"],
        "normalized_score": reward_result.get("normalized_score", 0.0),
        "task_success": reward_result["task_success"],
        "steps_taken": len(agent_steps),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "reward_result": reward_result,
    }
    episode_history.append(episode)
    return episode


def get_score_history() -> list:
    """Returns all past episode results (used for the improvement graph)."""
    return episode_history