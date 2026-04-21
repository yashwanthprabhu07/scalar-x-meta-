# ============================================================
# lesson_extractor.py — Distill a lesson from an episode
#
# After an episode ends, we ask Groq itself to look at what the
# agent did (tool trajectory), what the reward function caught
# (score, breakdown, state check results), and produce a short,
# actionable lesson that the agent can apply on future attempts
# of the same scenario.
#
# The lesson is prompt-level feedback — it's added to the system
# prompt for all future episodes of this scenario. This is how
# the Self-Improvement Graph actually goes up.
# ============================================================
import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


# ─────────────────────────────────────────────
# SYSTEM PROMPT FOR THE LESSON EXTRACTOR
# ─────────────────────────────────────────────
# We ask for output in a specific shape so we can parse it cleanly.
# Note we deliberately tell the model to return a SINGLE concise
# bullet — not a long postmortem — because this lesson will be
# stacked with others in the next episode's system prompt and we
# can't let them balloon.
_EXTRACTOR_SYSTEM_PROMPT = """\
You are a coaching assistant that reviews the work of another AI agent.
After each episode, you will be given:
  - The task the agent was supposed to perform
  - The sequence of tool calls the agent made (its trajectory)
  - The score the agent received
  - A breakdown of what was right and wrong (the reward breakdown)
  - A state check listing which real-world outcomes were achieved or missed

Your job is to produce ONE short, actionable lesson (1-2 sentences, max 40 words)
that, if the agent remembered it next time, would help it do better.

Rules:
  - Focus on the SINGLE most impactful mistake or insight from this episode.
    If the episode was perfect, produce a short positive reinforcement instead.
  - Be specific and concrete. Reference actual tool names or decisions.
    BAD:  "Be more careful when handling tasks."
    GOOD: "When reassigning an existing task, use assign_task with its task_id — do NOT call create_task, which creates a duplicate."
  - Write in imperative voice directed at the agent ("Use X", "Avoid Y", "Always Z").
  - Never exceed 2 sentences. Never write paragraphs.

Return ONLY the lesson text. No preamble, no quotes, no bullet markers, no explanation.
"""


def extract_lesson(
    scenario_name: str,
    agent_prompt: str,
    taken_actions: list,
    agent_steps: list,
    reward_result: dict,
    groq_client,
) -> str:
    """
    Use Groq to distill a one-sentence lesson from an episode.

    Args:
        scenario_name:   Human-readable scenario name (for context)
        agent_prompt:    The task the agent was told to perform
        taken_actions:   The sequence of tool names called (just names)
        agent_steps:     The full step dicts (includes inputs/results — we'll summarize)
        reward_result:   The dict from calculate_reward()
        groq_client:     A Groq client instance (already authenticated)

    Returns:
        A string lesson (may be empty if extraction failed — caller should handle)
    """

    # Build a compact trajectory summary. We don't want to send the full
    # tool_result JSON blobs — that's a token-burn. Just name + truncated input.
    trajectory_lines = []
    for i, step in enumerate(agent_steps, 1):
        name = step.get("tool_name", "?")
        inp = json.dumps(step.get("tool_input", {}), ensure_ascii=False)[:100]
        trajectory_lines.append(f"  {i}. {name}({inp})")
    trajectory = "\n".join(trajectory_lines) if trajectory_lines else "  (no tool calls made)"

    # Build the reward summary — just the breakdown bullets
    breakdown = "\n".join(f"  {line}" for line in reward_result.get("breakdown", []))

    score = reward_result.get("score", 0)
    max_score = reward_result.get("max_possible_score", 0)
    task_success = reward_result.get("task_success", False)
    state_ok = reward_result.get("state_check_passed", True)

    # The user message summarizes the episode for the coaching LLM
    user_message = f"""\
## Scenario: {scenario_name}
Task success: {task_success}
State check passed: {state_ok}
Score: {score} / {max_score}

## Agent's instructions (the task it was given):
{agent_prompt}

## Agent's tool call trajectory (what it actually did):
{trajectory}

## Reward breakdown (what was right and wrong):
{breakdown}

Based on the above, write ONE short actionable lesson (1-2 sentences, max 40 words)
that would help the agent do better next time on this exact scenario.
"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=150,
            messages=[
                {"role": "system", "content": _EXTRACTOR_SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
        )
        lesson = response.choices[0].message.content or ""
        lesson = lesson.strip()

        # Sanity caps — if Groq ignored our length instructions, trim hard.
        # This prevents a single rogue 500-word "lesson" from blowing up the
        # system prompt for every future episode.
        if len(lesson) > 400:
            lesson = lesson[:400].rsplit(".", 1)[0] + "."

        return lesson

    except Exception as e:
        # Lesson extraction is best-effort. If Groq fails here, we don't
        # want to kill the whole episode — just log and return empty.
        print(f"   ⚠️  Lesson extraction failed: {type(e).__name__}: {str(e)[:120]}")
        return ""