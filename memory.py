# ============================================================
# memory.py — Per-scenario "lessons learned" memory store
#
# After each episode, the agent distills a short lesson from its
# trajectory + reward and saves it here. Lessons are persisted
# to disk as JSON (keyed by scenario_id) so they survive across
# Streamlit restarts.
#
# At the start of the next episode of the SAME scenario, the
# accumulated lessons are injected into the system prompt so
# the agent can avoid repeating past mistakes.
#
# This is prompt-level self-improvement. It's simple, cheap,
# and most importantly: it actually changes agent behavior
# across episodes, which makes the Self-Improvement Graph
# meaningful instead of decorative.
# ============================================================
import json
import os
from datetime import datetime
from threading import Lock

# File where lessons are persisted. Gets created on first write.
MEMORY_FILE = "agent_memory.json"

# Guard against concurrent writes from the Streamlit thread + agent thread
_memory_lock = Lock()

# Maximum lessons to keep per scenario (oldest are dropped)
# We cap this so the system prompt doesn't grow unboundedly and
# burn tokens on lessons the agent has already internalized.
MAX_LESSONS_PER_SCENARIO = 5


def _load_all() -> dict:
    """Load the entire memory file. Returns empty dict if it doesn't exist."""
    if not os.path.exists(MEMORY_FILE):
        return {}
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # Corrupt or unreadable — start fresh rather than crashing
        return {}


def _save_all(data: dict) -> None:
    """Write the entire memory file (atomic-ish: write-then-rename)."""
    tmp = MEMORY_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, MEMORY_FILE)   # atomic on POSIX, best-effort on Windows


def get_lessons(scenario_id: str) -> list:
    """
    Return the list of lessons accumulated for this scenario, newest last.
    Each lesson is a dict with keys: text, episode_number, score, timestamp.
    """
    with _memory_lock:
        data = _load_all()
        return data.get(scenario_id, [])


def add_lesson(scenario_id: str, lesson_text: str, episode_number: int, score: int) -> None:
    """
    Append a new lesson for a scenario, capped at MAX_LESSONS_PER_SCENARIO.
    Silently no-ops if the lesson text is empty or whitespace-only.
    """
    lesson_text = (lesson_text or "").strip()
    if not lesson_text:
        return

    with _memory_lock:
        data = _load_all()
        lessons = data.get(scenario_id, [])
        lessons.append({
            "text": lesson_text,
            "episode_number": episode_number,
            "score": score,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        # Keep only the most recent N
        if len(lessons) > MAX_LESSONS_PER_SCENARIO:
            lessons = lessons[-MAX_LESSONS_PER_SCENARIO:]
        data[scenario_id] = lessons
        _save_all(data)


def format_lessons_for_prompt(scenario_id: str) -> str:
    """
    Turn the stored lessons into a prompt-ready string.
    Returns an empty string if there are no lessons yet.
    """
    lessons = get_lessons(scenario_id)
    if not lessons:
        return ""

    lines = [
        "",
        "── LESSONS FROM YOUR PAST ATTEMPTS ──",
        "Previous runs of this scenario taught you these things. "
        "Apply them this time to avoid repeating past mistakes:",
        "",
    ]
    for i, lesson in enumerate(lessons, 1):
        lines.append(f"{i}. {lesson['text']}")
    lines.append("")
    lines.append("── END OF LESSONS ──")
    return "\n".join(lines)


def clear_lessons(scenario_id: str = None) -> None:
    """
    Clear lessons for one scenario, or for all scenarios if scenario_id is None.
    Useful for the dashboard's 'Reset memory' button.
    """
    with _memory_lock:
        if scenario_id is None:
            _save_all({})
        else:
            data = _load_all()
            data.pop(scenario_id, None)
            _save_all(data)


def get_all_lessons() -> dict:
    """Return the full memory dict (for display in the dashboard)."""
    with _memory_lock:
        return _load_all()