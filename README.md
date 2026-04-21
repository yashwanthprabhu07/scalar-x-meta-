<!-- @format -->

# AI Enterprise Workflow Simulator

> An **OpenEnv-compatible evaluation harness and baseline agent** for long-horizon, multi-app enterprise workflows. Built for the **Scaler × Meta PyTorch OpenEnv Hackathon** (April 2026).

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-ff4b4b.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

---

## What this project is

Modern AI agents are expected to handle **long-horizon, multi-app workflows** — the kind a human employee does every day: read an email, update a CRM deal, book a meeting, post a Slack message, coordinate across five different tools to resolve one real-world situation.

Evaluating such agents is harder than evaluating chat responses. You can't just grade text output — you have to grade **outcomes**: did the deal actually get updated? Did the meeting actually get booked with the right people?

This project provides:

1. **A simulated enterprise environment** — 5 mock SaaS apps (Email, Chat, CRM, Tasks, Calendar) with 20 tools the agent can call.
2. **3 realistic long-horizon scenarios** — each requires 6–8 coordinated tool calls across multiple apps to succeed.
3. **A rigorous dual reward signal** — both _tool-call_ checks (did it use the right tools?) and _state-based_ checks (did the mock apps actually end up in the correct state?).
4. **A baseline tool-calling agent** (currently Llama-3.3-70B via Groq) plus a pluggable `AgentInterface` so any policy — including a PyTorch policy trained with OpenEnv — can be dropped in and scored the same way.
5. **A live dashboard** that streams agent actions in real time and plots reward over episodes.

> **Positioning note:** the repo ships a working baseline agent, but the real contribution is the **environment + reward + eval harness**. The agent is the easiest part to swap.

---

## Why it maps to the hackathon themes

| Theme                        | How this project addresses it                                                                                                                                                                                                    |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Long-Horizon Planning**    | Each scenario requires 6–8 interdependent tool calls across 5 apps. Success needs a plan, not a reflex.                                                                                                                          |
| **World Modeling**           | The agent must read state (inbox, channel, tasks) before acting. State-based checks verify the world was actually changed, not just that tools were invoked.                                                                     |
| **Self-Improving Agent**     | Between-episode memory: after each run, the agent distills lessons from its trajectory + reward and carries them into future episodes. The Self-Improvement Graph shows reward rising across repeated runs of the same scenario. |
| **Multi-Agent Interactions** | (Stretch) The environment supports multiple agents posting into shared chat channels and sharing task state; the baseline is single-agent but the substrate is multi-agent ready.                                                |

---

## Demo screenshots

_Add your own screenshots here after recording the demo video:_

- Sidebar with scenario selection and reward rules
- Live agent feed streaming tool calls in real time
- Reward breakdown with both tool-call and state-based checks
- Self-improvement graph showing score rising over episodes

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                    │
│        (dashboard.py — live feed + reward graph)          │
└────────────────────────┬─────────────────────────────────┘
                         │ progress callback
                         ▼
┌──────────────────────────────────────────────────────────┐
│                    Agent Loop  (agent.py)                 │
│   - Calls LLM via AgentInterface (pluggable backend)      │
│   - Dispatches tools via TOOL_DISPATCH table              │
│   - Retry-with-backoff on transient API errors            │
└───┬─────────────────────┬────────────────────┬───────────┘
    │                     │                    │
    ▼                     ▼                    ▼
┌──────────┐      ┌──────────────┐     ┌────────────────┐
│  Tools   │      │  Mock Apps   │     │     Reward     │
│(tools.py)│      │(mock_apps.py)│     │  (reward.py)   │
│          │      │              │     │                │
│ 20 tools │      │ Email, Chat, │     │ +10 success    │
│ schemas  │      │ CRM, Tasks,  │     │ +2 per correct │
│          │      │ Calendar     │     │ -1 per extra   │
│          │      │              │     │ -5 per missed  │
└──────────┘      └──────────────┘     │ + state checks │
                                        └────────────────┘
                         ▲
                         │
                  ┌──────┴──────┐
                  │  Scenarios  │
                  │(scenarios.py)│
                  │             │
                  │ 3 tasks +   │
                  │ state-based │
                  │ success     │
                  │ checks      │
                  └─────────────┘
```

### File map

| File           | Purpose                                                                               |
| -------------- | ------------------------------------------------------------------------------------- |
| `agent.py`     | The agent loop. Calls the LLM, dispatches tools, scores the episode.                  |
| `mock_apps.py` | The 5 simulated SaaS apps (Email, Chat, CRM, Tasks, Calendar).                        |
| `tools.py`     | JSON-schema definitions for all 20 tools exposed to the LLM.                          |
| `scenarios.py` | The 3 scenarios, each with a prompt, required tools, and a state-based success check. |
| `reward.py`    | The dual reward function + episode history logger.                                    |
| `dashboard.py` | The Streamlit dashboard (live feed, reward graph, episode history).                   |

---

## The 3 scenarios

### 🔥 Deal Rescue

A key client (Acme Corp, $50K/year) emails to cancel their contract. The agent must: read the email, look up the deal in CRM, reply with a discount offer, update the deal stage, add a note, book a follow-up call, and post to the sales channel. **8 required tool calls.**

### ⚔️ Team Conflict

Two engineers — Arjun and Sneha — both claim ownership of the same task. The agent must: read the engineering channel, inspect the conflicting task, list all tasks to find an alternative, **reassign** TASK-003 to Sneha and **reassign** TASK-004 to Arjun (the state check specifically catches agents that `create_task` instead of `assign_task` — a common LLM mistake), post a resolution, and book a sync meeting. **6 required tool calls.**

### 🚀 Client Onboarding

A new client (Priya Sharma, NewClient Inc) emails interest in the enterprise plan. The agent must: read the inbox (and correctly pick out Priya's email from among several), read the full message, create a CRM contact, reply with a welcome, create two onboarding tasks (one each for Arjun and Sneha), book a kickoff meeting, and announce it in the general channel. **7 required tool calls.**

---

## The reward function

The reward function has **two independent layers**:

**Layer 1 — Tool-call scoring** (what tools were called):

- `+10` if the task is fully completed
- `-10` if the task failed
- `+2` for each distinct required tool called correctly
- `-5` for each required tool that was missed
- `-1` for each extra call (duplicates of required tools beyond the first, or calls to non-required tools)

**Layer 2 — State-based verification** (what actually happened):
Each scenario defines a `success_check(apps)` that inspects the final state of the mock apps and returns a list of ✅/❌ outcomes. For example, Deal Rescue verifies that the Acme deal's stage actually changed, a new note was added, a new meeting exists, a reply was sent, and a message was posted.

**Overall success = all required tools called AND all state checks passed AND the agent didn't hit the iteration limit.** All three gates must pass.

This dual signal prevents agents from gaming the reward by, for example, calling `book_meeting` with garbage arguments — the tool call counts, but the state check catches that no valid meeting exists.

---

## Self-improving agent (Path B)

After each episode, the agent extracts a short "lesson learned" from its trajectory and final reward. On the next episode of the same scenario, the agent receives those lessons in its system prompt, allowing it to avoid past mistakes without any model retraining.

This is _prompt-level_ self-improvement — not reinforcement learning — but the **reward signal it uses is the same one a PyTorch policy would use**. When the environment is later plugged into OpenEnv, the existing reward function works unchanged.

Running the same scenario multiple times shows a visible upward trend in the Self-Improvement Graph.

---

## Pluggable agent backend (`agent_interface.py`)

The current baseline uses Llama-3.3-70B via the Groq API. The agent is decoupled from the environment through a small interface:

```python
class AgentInterface:
    def act(self, messages: list, tools: list) -> AgentResponse:
        """
        Given the conversation history and available tools,
        return either a tool call or a final text response.
        """
```

Anything implementing this interface can be plugged in — a different LLM, a local model, or a PyTorch policy wrapped in an OpenEnv-style adapter. The reward function, scenarios, and state checks all remain unchanged.

---

## Quickstart

### 1. Clone and set up

```bash
git clone https://github.com/yashwanthprabhu07/scalar-x-meta-.git
cd scalar-x-meta-
python -m venv venv
source venv/bin/activate          # or: venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Configure your LLM

Copy the example env file and add your Groq API key (free at [console.groq.com](https://console.groq.com)):

```bash
cp .env.example .env
# edit .env and set GROQ_API_KEY=your_key_here
```

### 3. Run the dashboard

```bash
streamlit run dashboard.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser, pick a scenario, click **▶ Run Agent**, and watch the live feed.

---

## Project status

This is a hackathon submission. It is **not** production code. Known limitations:

- In-memory state only (no persistence across dashboard restarts beyond session state).
- Groq free tier has a daily token limit; heavy testing can exhaust it.
- The self-improvement loop is prompt-based, not model-based.
- Only 3 scenarios are included; the architecture supports adding more trivially via `scenarios.py`.

---

## Future work

- **Swap baseline to Anthropic Claude** for production reliability.
- **Add a real PyTorch policy baseline** trained against this environment via OpenEnv.
- **Expand to 10+ scenarios** covering error recovery, partial-information tasks, and multi-agent coordination.
- **Add adversarial scenarios** — situations designed to trip common LLM failure modes (ID hallucination, unnecessary tool repetition, premature termination).

---

## Hackathon

Built for the **Scaler × Meta PyTorch OpenEnv Hackathon**, April 2026.

## License

MIT.
