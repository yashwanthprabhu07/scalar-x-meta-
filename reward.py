# ============================================================
# reward.py — Reward Function and Episode Logger
# This scores how well the agent did after each task.
# Higher score = agent did better.
# ============================================================
from collections import Counter
from datetime import datetime

# In-memory episode history (list of all past episode results)
episode_history = []


def calculate_reward(required_actions: list, taken_actions: list, task_success: bool) -> dict:
    """
    Scores the agent's performance for one episode.

    Scoring rules:
      +10 if task fully completed, else -10
      +2  for each DISTINCT required action that was correctly taken
      -5  for each required action that was MISSED
      -1  for every EXTRA call — this includes:
            * calls to tools not in required_actions
            * duplicate calls to required tools (first call is free, repeats cost -1 each)

    Args:
        required_actions: list of tool names the agent MUST call
        taken_actions:    list of tool names the agent actually called (in order)
        task_success:     True if all required actions were taken

    Returns:
        dict with score, breakdown, and metadata
    """
    score = 0
    breakdown = []

    required_set = set(required_actions)
    taken_counts = Counter(taken_actions)

    # +10 / -10 for task outcome
    if task_success:
        score += 10
        breakdown.append("+10 Task fully completed ✅")
    else:
        score -= 10
        breakdown.append("-10 Task failed ❌")

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
    #   * duplicate calls to required tools (count beyond the first)
    #   * every call to a non-required tool
    extra_count = 0
    for tool, count in taken_counts.items():
        if tool in required_set:
            # First call is free, every repeat is an extra
            extras = count - 1
            if extras > 0:
                extra_count += extras
                score -= extras
                breakdown.append(
                    f"-{extras} Duplicate call(s) to required tool: {tool} (called {count}x)"
                )
        else:
            # All calls to non-required tools are extras
            extra_count += count
            score -= count
            breakdown.append(
                f"-{count} Unnecessary tool call(s): {tool} (called {count}x)"
            )

    # Calculate max possible score for this scenario (for normalization later)
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
    """
    Saves the result of one episode to history.
    This history is used to plot the improvement graph.

    Args:
        scenario_id:    which scenario was run
        reward_result:  output from calculate_reward()
        agent_steps:    list of all steps the agent took

    Returns:
        the episode dict that was saved
    """
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