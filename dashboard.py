# ============================================================
# dashboard.py — Streamlit Dashboard
# Run with: streamlit run dashboard.py
# ============================================================

import streamlit as st
import threading
import queue
import time

from scenarios import SCENARIOS, get_scenario_instance
from agent import run_agent

# Module-level queue. Background worker writes here; the Streamlit
# main thread drains it on every rerun. Keeping it at module scope
# avoids the well-known "session_state can't share queues across
# rerun cycles cleanly" pitfall.
_STEP_QUEUE: queue.Queue = queue.Queue()


# Tool -> accent color, for live-feed visual grouping.
_TOOL_ACCENTS = (
    (("read_inbox", "read_email", "send_email", "reply_email"),               "#58a6ff"),
    (("list_channels", "read_channel", "post_message"),                       "#d2a8ff"),
    (("get_deal", "update_deal_stage", "add_note", "get_contact",
      "create_contact"),                                                      "#f78166"),
    (("list_tasks", "get_task", "create_task", "assign_task", "close_task"),  "#7ee787"),
    (("list_meetings", "check_conflicts", "book_meeting"),                    "#ffd166"),
)


def _tool_color(name: str) -> str:
    for prefixes, color in _TOOL_ACCENTS:
        if name in prefixes:
            return color
    return "#58a6ff"


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Enterprise Workflow Simulator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
  --bg:      #0a0e14;
  --panel:   #11161e;
  --panel-2: #161b22;
  --line:    #232a34;
  --text:    #e6edf3;
  --mute:    #8b949e;
  --blue:    #58a6ff;
  --green:   #7ee787;
  --orange:  #f78166;
  --purple:  #d2a8ff;
  --yellow:  #ffd166;
  --red:     #f87171;
}

.stApp {
  background:
    radial-gradient(circle at 12% -5%, rgba(88,166,255,0.10) 0%, transparent 35%),
    radial-gradient(circle at 88% 0%,  rgba(247,129,102,0.08) 0%, transparent 40%),
    var(--bg);
  color: var(--text);
}

section[data-testid="stSidebar"] {
  background-color: var(--panel) !important;
  border-right: 1px solid var(--line);
}

/* Hero */
.hero-wrap { display:flex; align-items:center; gap:1rem; padding: 0.4rem 0 0.2rem 0; }
.hero-mark {
  width:44px; height:44px; border-radius:11px;
  background: linear-gradient(135deg, #1f6feb, #388bfd 60%, #7ee787);
  display:flex; align-items:center; justify-content:center;
  font-size:1.5rem;
  box-shadow: 0 0 28px rgba(88,166,255,.45), inset 0 0 18px rgba(255,255,255,.07);
}
.hero-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.95rem; font-weight: 700; letter-spacing: -0.5px;
  background: linear-gradient(90deg, #7ee787 0%, #58a6ff 50%, #f78166 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  margin: 0; line-height: 1.1;
}
.hero-sub {
  font-family: 'JetBrains Mono', monospace;
  color: var(--mute); font-size: 0.78rem; margin-top: 0.15rem;
}
.hero-pills { display:flex; gap:0.4rem; flex-wrap:wrap; margin-top: 0.6rem; }
.pill {
  display:inline-flex; align-items:center; gap:0.35rem;
  font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
  padding: 0.22rem 0.6rem; border-radius: 999px;
  background: rgba(88,166,255,0.08); color:#9fc6ff;
  border: 1px solid rgba(88,166,255,0.25);
}
.pill.green  { background: rgba(126,231,135,0.08); color:#aff0b3; border-color: rgba(126,231,135,0.25); }
.pill.orange { background: rgba(247,129,102,0.08); color:#ffb39a; border-color: rgba(247,129,102,0.30); }
.pill.purple { background: rgba(210,168,255,0.08); color:#e1c3ff; border-color: rgba(210,168,255,0.30); }

/* Section headers */
.section-h {
  display:flex; align-items:center; gap:0.55rem;
  margin: 0.6rem 0 0.55rem 0;
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.05rem; font-weight: 600; letter-spacing: 0.2px;
  color: var(--text);
}
.section-h .dot {
  width:8px; height:8px; border-radius: 50%; background: var(--blue);
  box-shadow: 0 0 10px var(--blue);
}

/* Cards */
.card {
  background: var(--panel-2);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 0.9rem 1rem;
}
.card.tight { padding: 0.6rem 0.8rem; }

/* Live step card */
.step-card {
  background: var(--panel-2);
  border: 1px solid var(--line);
  border-left: 3px solid var(--blue);
  border-radius: 8px;
  padding: 0.65rem 0.9rem;
  margin-bottom: 0.55rem;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem;
  transition: transform .12s ease, border-color .15s ease;
}
.step-card:hover { transform: translateX(2px); border-color: rgba(88,166,255,0.6); }
.step-num    { color: var(--mute); font-size: 0.7rem; }
.step-tool   { font-weight: 700; }
.step-args   { color: #cdd9e5; margin-top: 4px; font-size: 0.72rem; word-break: break-all; }
.step-result { color: #aff0b3; margin-top: 3px; font-size: 0.72rem; word-break: break-all; }
.step-result.err { color: #ff9f9f; }

/* Score badge */
.score-badge {
  display:inline-flex; align-items:center; gap:0.5rem;
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.7rem; font-weight: 700;
  padding: 0.55rem 1.3rem; border-radius: 10px;
  margin: 0.4rem 0; color: white;
}
.score-positive { background: linear-gradient(135deg, #196c2e, #2ea043); box-shadow: 0 0 32px rgba(46,160,67,0.35); }
.score-negative { background: linear-gradient(135deg, #6e2316, #b62324); box-shadow: 0 0 32px rgba(182,35,36,0.30); }
.score-zero     { background: linear-gradient(135deg, #3a3f47, #5c6370); }

/* Metric cards */
.metric-card {
  background: var(--panel-2);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 0.9rem 0.5rem;
  text-align: center;
}
.metric-val {
  font-size: 1.55rem; font-weight: 700;
  font-family: 'Space Grotesk', sans-serif; line-height: 1;
}
.metric-lbl {
  font-size: 0.7rem; color: var(--mute);
  font-family: 'JetBrains Mono', monospace;
  margin-top: 0.3rem;
  text-transform: uppercase; letter-spacing: 0.6px;
}

/* Status pulse */
.status-running {
  display:inline-flex; align-items:center; gap:0.55rem;
  background: rgba(88,166,255,0.10);
  border: 1px solid rgba(88,166,255,0.35);
  color: #cfe3ff;
  padding: 0.35rem 0.75rem; border-radius: 999px;
  font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
  margin-bottom: 0.6rem;
}
.pulse-dot {
  width: 8px; height: 8px; border-radius: 50%; background: var(--blue);
  box-shadow: 0 0 10px var(--blue);
  animation: pulse 1.2s ease-in-out infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 0.4; transform: scale(0.85); }
  50%      { opacity: 1.0; transform: scale(1.10); }
}

/* Reward breakdown rows */
.rb-row {
  display:flex; align-items:flex-start; gap:0.45rem;
  font-family: 'JetBrains Mono', monospace; font-size: 0.76rem;
  padding: 0.22rem 0;
}
.rb-row .tag { width: 1.2rem; flex: 0 0 auto; }
.rb-pos     { color: #aff0b3; }
.rb-neg     { color: #ff9f9f; }
.rb-neutral { color: var(--mute); }

/* Lesson card */
.lesson-card {
  border: 1px dashed rgba(210,168,255,0.45);
  background: rgba(210,168,255,0.06);
  border-radius: 10px; padding: 0.7rem 0.9rem;
  font-family: 'Inter', sans-serif; font-size: 0.84rem;
  color: #e1c3ff;
}

/* Buttons */
.stButton > button {
  background: linear-gradient(135deg, #1f6feb, #388bfd);
  color: white; border: none; border-radius: 8px;
  font-family: 'Space Grotesk', sans-serif; font-weight: 600;
  padding: 0.55rem 1.5rem; width: 100%;
  transition: transform .1s ease, box-shadow .15s ease;
  box-shadow: 0 4px 18px rgba(31,111,235,0.25);
}
.stButton > button:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 6px 24px rgba(31,111,235,0.4);
}
.stButton > button:disabled {
  background: #232a34; color: #8b949e; box-shadow: none;
}

footer    { visibility: hidden; }
#MainMenu { visibility: hidden; }

.block-container { padding-top: 1.2rem !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
def _init_state():
    defaults = {
        "running":         False,
        "live_steps":      [],
        "last_result":     None,
        "episode_history": [],
        "active_scenario": None,
        "started_at":      None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ─────────────────────────────────────────────
# THREAD HELPERS
# ─────────────────────────────────────────────
def _drain_queue():
    while True:
        try:
            item = _STEP_QUEUE.get_nowait()
        except queue.Empty:
            return

        if "__done__" in item:
            st.session_state.running = False
            if item.get("__error__"):
                st.session_state.last_result = {"__error__": item["__error__"]}
            else:
                result = item["result"]
                st.session_state.last_result = result
                ep = result["episode"]
                st.session_state.episode_history.append({
                    "episode_number": len(st.session_state.episode_history) + 1,
                    "scenario_id":    ep["scenario_id"],
                    "score":          ep["score"],
                    "task_success":   ep["task_success"],
                    "steps_taken":    ep["steps_taken"],
                    "timestamp":      ep["timestamp"],
                })
        else:
            st.session_state.live_steps.append(item)


def _start_agent_run(scenario: dict):
    """Reset per-run state and launch the agent in a background thread."""
    st.session_state.running         = True
    st.session_state.live_steps      = []
    st.session_state.last_result     = None
    st.session_state.active_scenario = scenario
    st.session_state.started_at      = time.time()

    while not _STEP_QUEUE.empty():
        try:
            _STEP_QUEUE.get_nowait()
        except queue.Empty:
            break

    def _worker():
        def on_step(step):
            _STEP_QUEUE.put(step)

        try:
            result = run_agent(scenario=scenario, progress_callback=on_step)
            _STEP_QUEUE.put({"__done__": True, "result": result})
        except Exception as e:
            _STEP_QUEUE.put({"__done__": True, "result": None, "__error__": str(e)})

    threading.Thread(target=_worker, daemon=True).start()


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown(
    """
<div class="hero-wrap">
  <div class="hero-mark">🤖</div>
  <div>
    <div class="hero-title">AI Enterprise Workflow Simulator</div>
    <div class="hero-sub">Meta PyTorch OpenEnv × Scaler · Bangalore · Apr 2026</div>
  </div>
</div>
<div class="hero-pills">
  <span class="pill">OpenEnv 0.2.3</span>
  <span class="pill green">TRL · GRPO</span>
  <span class="pill orange">Self-Improving</span>
  <span class="pill purple">5 apps · 20 tools</span>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Control Panel")
    st.markdown("---")

    scenario_names    = [s["name"] for s in SCENARIOS]
    selected_name     = st.selectbox(
        "Scenario",
        scenario_names,
        disabled=st.session_state.running,
    )
    selected_template = next(s for s in SCENARIOS if s["name"] == selected_name)

    st.markdown("**Description**")
    st.info(selected_template["description"])

    st.markdown("**Required Actions**")
    actions_html = "".join(
        f"<span class='pill' style='margin:2px 4px 2px 0;font-size:.65rem'>{a}</span>"
        for a in selected_template["required_actions"]
    )
    st.markdown(actions_html, unsafe_allow_html=True)

    st.markdown("---")

    run_clicked   = st.button(
        "▶ Run Agent" if not st.session_state.running else "⏳ Agent Running…",
        disabled=st.session_state.running,
    )
    rerun_clicked = False
    if st.session_state.last_result and not st.session_state.running:
        rerun_clicked = st.button("🔄 Run Again (fresh randomized episode)")

    st.markdown("---")
    st.markdown("**Reward Rules**")
    st.markdown(
        """
- ✅ Task complete: **+10**
- 🔧 Correct tool: **+2**
- ➕ Extra step: **−1**
- ❌ Missed required: **−5**
- 💀 Task failed: **−10**
        """
    )

    st.markdown("---")
    st.markdown("**Hackathon Themes**")
    for theme in [
        "Long-Horizon Planning",
        "World Modeling / Pro Tasks",
        "Self-Improving Agent",
        "Multi-App Coordination",
    ]:
        st.markdown(f"• {theme}")

    st.markdown("---")
    if st.button("🗑 Clear History", disabled=st.session_state.running):
        st.session_state.episode_history = []
        st.session_state.last_result     = None
        st.session_state.live_steps      = []
        st.session_state.active_scenario = None
        st.rerun()


# ─────────────────────────────────────────────
# TRIGGER RUNS
# ─────────────────────────────────────────────
# Each trigger generates a fresh randomized scenario instance, so consecutive
# runs of the same scenario template see different clients, amounts, etc.
if run_clicked:
    _start_agent_run(get_scenario_instance(selected_template["id"]))
elif rerun_clicked:
    _start_agent_run(get_scenario_instance(selected_template["id"]))


# Drain anything the worker has produced since the last rerun.
_drain_queue()


# ─────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

# ── LEFT: Live Agent Feed ─────────────────────
with col_left:
    st.markdown(
        '<div class="section-h"><span class="dot"></span>Live Agent Feed</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.running:
        elapsed = int(time.time() - (st.session_state.started_at or time.time()))
        st.markdown(
            f"""<div class='status-running'>
              <span class='pulse-dot'></span>
              Agent thinking · {len(st.session_state.live_steps)} step(s) · {elapsed}s elapsed
            </div>""",
            unsafe_allow_html=True,
        )

    steps  = st.session_state.live_steps
    active = st.session_state.active_scenario

    # Progress against required actions for the *currently active* (randomized) scenario
    if active and (st.session_state.running or steps):
        required_unique = list(dict.fromkeys(active["required_actions"]))
        taken_set       = {s.get("tool_name") for s in steps}
        done_n          = sum(1 for r in required_unique if r in taken_set)
        st.progress(
            min(done_n / max(1, len(required_unique)), 1.0),
            text=f"Required actions called: {done_n}/{len(required_unique)}",
        )

        with st.expander("📜 Agent prompt for this episode", expanded=False):
            st.code(active["agent_prompt"], language="markdown")

    if steps:
        steps_html = ""
        for i, step in enumerate(steps, 1):
            tool_name   = step.get("tool_name", "")
            tool_input  = str(step.get("tool_input", {}))[:160]
            tool_result = str(step.get("tool_result", ""))[:260]
            color       = _tool_color(tool_name)
            err_class   = "err" if '"error"' in tool_result.lower() else ""
            steps_html += f"""
            <div class="step-card" style="border-left-color:{color}">
              <span class="step-num">Step {i:02d}</span>
              &nbsp;→&nbsp;
              <span class="step-tool" style="color:{color}">🔧 {tool_name}</span>
              <div class="step-args">{tool_input}</div>
              <div class="step-result {err_class}">↳ {tool_result}</div>
            </div>
            """
        st.markdown(steps_html, unsafe_allow_html=True)
    elif not st.session_state.running:
        st.markdown(
            """<div class='card' style='text-align:center;color:var(--mute);padding:2rem 1rem;'>
              <div style='font-size:2rem;margin-bottom:.4rem'>📡</div>
              <div style='font-family:JetBrains Mono;font-size:.85rem'>
                Pick a scenario and hit <b style='color:#9fc6ff'>▶ Run Agent</b> to start.
              </div>
            </div>""",
            unsafe_allow_html=True,
        )


# ── RIGHT: Episode Results ────────────────────
with col_right:
    st.markdown(
        '<div class="section-h"><span class="dot" style="background:#7ee787;box-shadow:0 0 10px #7ee787"></span>Episode Results</div>',
        unsafe_allow_html=True,
    )

    result = st.session_state.last_result

    if result and result.get("__error__"):
        st.error(f"❌ Agent error: {result['__error__']}")

    elif result:
        reward  = result["reward"]
        score   = reward["score"]
        success = reward["task_success"]

        if score > 0:
            badge_class = "score-positive"
        elif score < 0:
            badge_class = "score-negative"
        else:
            badge_class = "score-zero"

        st.markdown(
            f'<div class="score-badge {badge_class}">Score: {score:+d}</div>',
            unsafe_allow_html=True,
        )
        if success:
            st.success("✅ TASK SUCCEEDED")
        else:
            st.error("❌ TASK FAILED")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f"""<div class='metric-card'>
                  <div class='metric-val' style='color:#58a6ff'>{len(result['steps'])}</div>
                  <div class='metric-lbl'>steps taken</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with m2:
            required = set(reward["required_actions"])
            taken    = set(reward["taken_actions"])
            correct  = len(required & taken)
            st.markdown(
                f"""<div class='metric-card'>
                  <div class='metric-val' style='color:#7ee787'>{correct}/{len(required)}</div>
                  <div class='metric-lbl'>required hit</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with m3:
            total = len(st.session_state.episode_history)
            st.markdown(
                f"""<div class='metric-card'>
                  <div class='metric-val' style='color:#f78166'>{total}</div>
                  <div class='metric-lbl'>episodes</div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.markdown("**Reward Breakdown**")

        rb_html = ""
        for line in reward["breakdown"]:
            line_s = line.strip()
            if not line_s:
                rb_html += "<div style='height:.4rem'></div>"
            # Section headers start with "-- " or "── " (ASCII vs unicode dashes)
            # — must be checked BEFORE the single-dash branch.
            elif line_s.startswith("-- ") or line_s.startswith("── "):
                rb_html += f"<div class='rb-row rb-neutral' style='margin-top:.3rem'><span>{line_s}</span></div>"
            elif line_s.startswith("+"):
                rb_html += f"<div class='rb-row rb-pos'><span class='tag'>🟢</span><span>{line_s}</span></div>"
            elif line_s.startswith("-"):
                rb_html += f"<div class='rb-row rb-neg'><span class='tag'>🔴</span><span>{line_s}</span></div>"
            else:
                rb_html += f"<div class='rb-row rb-neutral'><span class='tag'>•</span><span>{line_s}</span></div>"
        st.markdown(f"<div class='card tight'>{rb_html}</div>", unsafe_allow_html=True)

        new_lesson = reward.get("new_lesson")
        if new_lesson:
            st.markdown("&nbsp;", unsafe_allow_html=True)
            st.markdown("**🎓 Lesson Saved for Next Episode**")
            st.markdown(
                f"<div class='lesson-card'>{new_lesson}</div>",
                unsafe_allow_html=True,
            )

    else:
        st.markdown(
            """<div class='card' style='text-align:center;color:var(--mute);padding:2rem 1rem;'>
              <div style='font-size:2rem;margin-bottom:.4rem'>🏆</div>
              <div style='font-family:JetBrains Mono;font-size:.85rem'>
                Score, reward breakdown, and saved lessons appear here after each run.
              </div>
            </div>""",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
# IMPROVEMENT GRAPH
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div class="section-h"><span class="dot" style="background:#f78166;box-shadow:0 0 10px #f78166"></span>Self-Improvement Over Episodes</div>',
    unsafe_allow_html=True,
)

history = st.session_state.episode_history

if len(history) >= 1:
    import pandas as pd
    import plotly.express as px

    df = pd.DataFrame(
        [
            {
                "Episode":  ep["episode_number"],
                "Score":    ep["score"],
                "Scenario": ep["scenario_id"].replace("scenario_", "").replace("_", " ").title(),
            }
            for ep in history
        ]
    )

    fig = px.line(
        df, x="Episode", y="Score", color="Scenario", markers=True,
        labels={"Score": "Reward Score", "Episode": "Episode #"},
        template="plotly_dark",
        color_discrete_sequence=["#58a6ff", "#7ee787", "#f78166", "#d2a8ff"],
    )
    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=10, line=dict(width=1, color="#0a0e14")),
    )
    fig.update_layout(
        paper_bgcolor="#0a0e14", plot_bgcolor="#11161e",
        font=dict(family="JetBrains Mono", color="#e6edf3"),
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(bgcolor="#11161e", bordercolor="#232a34", borderwidth=1),
        xaxis=dict(gridcolor="#232a34", zerolinecolor="#232a34"),
        yaxis=dict(gridcolor="#232a34", zerolinecolor="#232a34"),
        height=360,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#8b949e", annotation_text="zero")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.markdown(
        """<div class='card' style='text-align:center;color:var(--mute);padding:1.6rem 1rem;border-style:dashed'>
          ▶ Run at least one scenario to see the improvement graph here.
        </div>""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# HISTORY TABLE
# ─────────────────────────────────────────────
if len(history) >= 1:
    st.markdown(
        '<div class="section-h"><span class="dot" style="background:#d2a8ff;box-shadow:0 0 10px #d2a8ff"></span>Episode History</div>',
        unsafe_allow_html=True,
    )
    import pandas as pd
    df_hist = pd.DataFrame(
        [
            {
                "Episode":   ep["episode_number"],
                "Scenario":  ep["scenario_id"].replace("scenario_", "").replace("_", " ").title(),
                "Score":     ep["score"],
                "Success":   "✅" if ep["task_success"] else "❌",
                "Steps":     ep["steps_taken"],
                "Timestamp": ep["timestamp"],
            }
            for ep in history
        ]
    )
    st.dataframe(df_hist, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
<div style='text-align:center;color:#8b949e;font-family:JetBrains Mono;font-size:0.72rem;
            padding:0.5rem 0 1.5rem 0;'>
  Yashwanth Prabhu R · Vivek Gowda NV · Chaitanya S Shetty &nbsp;·&nbsp;
  <a href='https://huggingface.co/spaces/yashwanthprabhu/enterprise-workflow-env' style='color:#58a6ff;text-decoration:none'>HF Space</a>
  &nbsp;·&nbsp;
  <a href='https://github.com/yashwanthprabhu07/scalar-x-meta-' style='color:#58a6ff;text-decoration:none'>GitHub</a>
</div>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# AUTO-REFRESH WHILE RUNNING
# ─────────────────────────────────────────────
if st.session_state.running:
    time.sleep(0.4)
    st.rerun()
