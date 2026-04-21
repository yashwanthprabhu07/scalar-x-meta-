# ============================================================
# reward.py — Reward Function and Episode Logger
# This scores how well the agent did after each task.
# Higher score = agent did better.
# ============================================================

import json
from datetime import datetime

# In-memory episode history (list of all past episode results)
episode_history = []


def calculate_reward(required_actions: list, taken_actions: list, task_success: bool) -> dict:
    """
    Scores the agent's performance for one episode.

    Args:
        required_actions: list of tool names the agent MUST call
        taken_actions:    list of tool names the agent actually called
        task_success:     True if all required actions were taken

    Returns:
        dict with score, breakdown, and metadata
    """
    score = 0
    breakdown = []

    required_set = set(required_actions)
    taken_set    = set(taken_actions)

    # +10 if task fully completed
    if task_success:
        score += 10
        breakdown.append("+10  Task fully completed ✅")
    else:
        score -= 10
        breakdown.append("-10  Task failed ❌")

    # +2 for each required action that was correctly taken
    correct_actions = required_set & taken_set
    for action in correct_actions:
        score += 2
        breakdown.append(f"+2   Correct tool used: {action}")

    # -1 for each unnecessary extra step (tools called but not required)
    extra_actions = [a for a in taken_actions if a not in required_set]
    for action in extra_actions:
        score -= 1
        breakdown.append(f"-1   Extra step taken: {action}")

    # -5 for each required action that was MISSED
    missed_actions = required_set - taken_set
    for action in missed_actions:
        score -= 5
        breakdown.append(f"-5   Required action missed: {action}")

    return {
        "score": score,
        "task_success": task_success,
        "breakdown": breakdown,
        "required_actions": list(required_actions),
        "taken_actions": taken_actions,
        "correct_count": len(correct_actions),
        "missed_count": len(missed_actions),
        "extra_count": len(extra_actions),
    }


def log_episode(scenario_id: str, reward_result: dict, agent_steps: list) -> dict:
    """
    Saves the result of one episode to history.
    This history is used to plot the improvement graph.

    Args:
        scenario_id:   which scenario was run
        reward_result: output from calculate_reward()
        agent_steps:   list of all steps the agent took

    Returns:
        the episode dict that was saved
    """
    episode = {
        "episode_number": len(episode_history) + 1,
        "scenario_id":    scenario_id,
        "score":          reward_result["score"],
        "task_success":   reward_result["task_success"],
        "steps_taken":    len(agent_steps),
        "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "reward_result":  reward_result,
    }
    episode_history.append(episode)
    return episode


def get_score_history() -> list:
    """Returns all past episode results (used for the improvement graph)."""
    return episode_history
