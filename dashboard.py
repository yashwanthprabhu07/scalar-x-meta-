# ============================================================
# dashboard.py — Streamlit Dashboard (Final Fixed Version)
# Run with: streamlit run dashboard.py
# ============================================================

import streamlit as st
import threading
import queue
import time

from scenarios import SCENARIOS
from agent import run_agent

# ─────────────────────────────────────────────
# GLOBAL QUEUE — stored at module level, NOT in session_state
# Background threads can access this directly
# ─────────────────────────────────────────────
_STEP_QUEUE = queue.Queue()

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
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Space+Grotesk:wght@400;600;700&display=swap');
    .stApp { background-color: #0d1117; color: #e6edf3; }
    section[data-testid="stSidebar"] { background-color: #161b22 !important; }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(90deg, #58a6ff, #7ee787, #f78166);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        font-family: 'JetBrains Mono', monospace;
        color: #8b949e; font-size: 0.85rem; margin-bottom: 1.5rem;
    }
    .step-card {
        background: #161b22; border: 1px solid #30363d;
        border-left: 3px solid #58a6ff; border-radius: 6px;
        padding: 0.8rem 1rem; margin-bottom: 0.6rem;
        font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
    }
    .score-badge {
        display: inline-block; color: white;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem; font-weight: 700;
        padding: 0.4rem 1.2rem; border-radius: 8px; margin: 0.5rem 0;
    }
    .score-positive { background: linear-gradient(135deg, #196c2e, #2ea043); }
    .score-negative { background: linear-gradient(135deg, #6e2316, #b62324); }
    .metric-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 8px; padding: 1rem; text-align: center;
    }
    .metric-val { font-size: 1.6rem; font-weight: 700; font-family: 'Space Grotesk', sans-serif; }
    .metric-lbl { font-size: 0.75rem; color: #8b949e; font-family: 'JetBrains Mono', monospace; }
    .stButton > button {
        background: linear-gradient(135deg, #1f6feb, #388bfd);
        color: white; border: none; border-radius: 6px;
        font-family: 'Space Grotesk', sans-serif; font-weight: 600;
        padding: 0.5rem 1.5rem; width: 100%;
    }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE — only simple values here, NO queue
# ─────────────────────────────────────────────
if "running"         not in st.session_state: st.session_state.running         = False
if "live_steps"      not in st.session_state: st.session_state.live_steps      = []
if "last_result"     not in st.session_state: st.session_state.last_result     = None
if "episode_history" not in st.session_state: st.session_state.episode_history = []

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="hero-title">🤖 AI Enterprise Workflow Simulator</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Meta PyTorch OpenEnv Hackathon · Scaler School of Technology · Apr 2026</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Control Panel")
    st.markdown("---")

    scenario_names    = [s["name"] for s in SCENARIOS]
    selected_name     = st.selectbox("Choose a scenario:", scenario_names)
    selected_scenario = next(s for s in SCENARIOS if s["name"] == selected_name)

    st.markdown("**What happens:**")
    st.info(selected_scenario["description"])

    st.markdown("**Required actions:**")
    for action in selected_scenario["required_actions"]:
        st.markdown(f"• `{action}`")

    st.markdown("---")
    run_clicked = st.button("▶ Run Agent", disabled=st.session_state.running)

    st.markdown("---")
    st.markdown("**Reward Rules:**")
    st.markdown("""
- ✅ Task complete: **+10**
- 🔧 Correct tool: **+2**
- ➕ Extra step: **−1**
- ❌ Missed required: **−5**
- 💀 Task failed: **−10**
    """)
    st.markdown("---")
    st.markdown("**Hackathon Themes:**")
    for theme in ["Multi-Agent Interactions", "Long-Horizon Planning", "World Modeling", "Self-Improving Agent"]:
        st.markdown(f"• {theme}")
    st.markdown("---")
    if st.button("🗑 Clear History"):
        st.session_state.episode_history = []
        st.session_state.last_result     = None
        st.session_state.live_steps      = []
        st.rerun()

# ─────────────────────────────────────────────
# START AGENT THREAD WHEN RUN IS CLICKED
# ─────────────────────────────────────────────
if run_clicked:
    st.session_state.running     = True
    st.session_state.live_steps  = []
    st.session_state.last_result = None

    # Clear the global queue
    while not _STEP_QUEUE.empty():
        try: _STEP_QUEUE.get_nowait()
        except queue.Empty: break

    # Capture scenario now (before thread starts)
    scenario_for_thread = selected_scenario

    def agent_thread_fn():
        # Use the GLOBAL queue — not session_state
        def on_step(step):
            _STEP_QUEUE.put(step)

        try:
            result = run_agent(scenario=scenario_for_thread, progress_callback=on_step)
            _STEP_QUEUE.put({"__done__": True, "result": result})
        except Exception as e:
            _STEP_QUEUE.put({"__done__": True, "result": None, "__error__": str(e)})

    threading.Thread(target=agent_thread_fn, daemon=True).start()

# ─────────────────────────────────────────────
# DRAIN QUEUE ON EVERY RERUN
# ─────────────────────────────────────────────
while True:
    try:
        item = _STEP_QUEUE.get_nowait()
    except queue.Empty:
        break

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

# Keep refreshing while running
if st.session_state.running:
    time.sleep(0.4)
    st.rerun()

# ─────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

# ── LEFT: Live Agent Feed ─────────────────────
with col_left:
    st.markdown("### 📡 Live Agent Feed")

    if st.session_state.running:
        st.info("🔄 Agent is working...")

    steps = st.session_state.live_steps
    if steps:
        steps_html = ""
        for i, step in enumerate(steps, 1):
            tool_name   = step.get("tool_name", "")
            tool_input  = str(step.get("tool_input", {}))[:150]
            tool_result = str(step.get("tool_result", ""))[:250]
            steps_html += f"""
            <div class="step-card">
                <span style="color:#8b949e">Step {i}</span> &nbsp;→&nbsp;
                <span style="color:#79c0ff;font-weight:700">🔧 {tool_name}</span>
                <div style="color:#cdd9e5;margin-top:4px;font-size:0.73rem">{tool_input}</div>
                <div style="color:#7ee787;margin-top:3px">↳ {tool_result}</div>
            </div>
            """
        st.markdown(steps_html, unsafe_allow_html=True)
    elif not st.session_state.running:
        st.markdown(
            "<div style='color:#8b949e;font-family:JetBrains Mono;font-size:0.85rem'>"
            "Select a scenario and click ▶ Run Agent to start.</div>",
            unsafe_allow_html=True,
        )

# ── RIGHT: Episode Results ────────────────────
with col_right:
    st.markdown("### 🏆 Episode Results")
    result = st.session_state.last_result

    if result and result.get("__error__"):
        st.error(f"❌ Error: {result['__error__']}")

    elif result:
        reward  = result["reward"]
        score   = reward["score"]
        success = reward["task_success"]

        badge_class = "score-positive" if score >= 0 else "score-negative"
        st.markdown(
            f'<div class="score-badge {badge_class}">Score: {score:+d}</div>',
            unsafe_allow_html=True,
        )
        if success:
            st.success("✅ TASK SUCCEEDED")
        else:
            st.error("❌ TASK FAILED")

        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-val" style="color:#58a6ff">{len(result['steps'])}</div>
                <div class="metric-lbl">steps taken</div></div>""", unsafe_allow_html=True)
        with m2:
            required = set(reward["required_actions"])
            taken    = set(reward["taken_actions"])
            correct  = len(required & taken)
            st.markdown(f"""<div class="metric-card">
                <div class="metric-val" style="color:#7ee787">{correct}/{len(required)}</div>
                <div class="metric-lbl">required actions</div></div>""", unsafe_allow_html=True)
        with m3:
            total = len(st.session_state.episode_history)
            st.markdown(f"""<div class="metric-card">
                <div class="metric-val" style="color:#f78166">{total}</div>
                <div class="metric-lbl">total episodes</div></div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Reward Breakdown:**")
        for line in reward["breakdown"]:
            if line.startswith("+"):
                st.markdown(f"🟢 `{line}`")
            elif line.startswith("-"):
                st.markdown(f"🔴 `{line}`")
            else:
                st.markdown(f"⚪ {line}")
    else:
        st.markdown(
            "<div style='color:#8b949e;font-family:JetBrains Mono;font-size:0.85rem'>"
            "Results will appear here after the agent runs.</div>",
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────
# SELF-IMPROVEMENT GRAPH
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📈 Self-Improvement Graph — Score Over Episodes")

history = st.session_state.episode_history

if len(history) >= 1:
    import pandas as pd
    import plotly.express as px

    df = pd.DataFrame([
        {
            "Episode":  ep["episode_number"],
            "Score":    ep["score"],
            "Scenario": ep["scenario_id"].replace("scenario_", "").replace("_", " ").title(),
        }
        for ep in history
    ])

    fig = px.line(
        df, x="Episode", y="Score", color="Scenario", markers=True,
        title="Agent Reward Improvement Over Time",
        labels={"Score": "Reward Score", "Episode": "Episode #"},
        template="plotly_dark",
        color_discrete_sequence=["#58a6ff", "#7ee787", "#f78166"],
    )
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font=dict(family="JetBrains Mono", color="#e6edf3"),
        title_font_size=16,
        legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#8b949e", annotation_text="Zero baseline")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.markdown(
        "<div style='color:#8b949e;font-family:JetBrains Mono;font-size:0.85rem;"
        "padding:2rem;text-align:center;border:1px dashed #30363d;border-radius:8px;'>"
        "▶ Run at least one scenario to see the improvement graph here.</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# EPISODE HISTORY TABLE
# ─────────────────────────────────────────────
if len(history) >= 1:
    st.markdown("---")
    st.markdown("### 📋 Episode History")
    import pandas as pd
    df_hist = pd.DataFrame([
        {
            "Episode":   ep["episode_number"],
            "Scenario":  ep["scenario_id"].replace("scenario_", "").replace("_", " ").title(),
            "Score":     ep["score"],
            "Success":   "✅" if ep["task_success"] else "❌",
            "Steps":     ep["steps_taken"],
            "Timestamp": ep["timestamp"],
        }
        for ep in history
    ])
    st.dataframe(df_hist, use_container_width=True, hide_index=True)
