---
title: AI Enterprise Workflow Environment
emoji: 🏢
colorFrom: pink
colorTo: pink
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# AI Enterprise Workflow Environment

> Train LLM agents to handle real enterprise workflows across 5 apps and 20 tools. Built for the Meta PyTorch OpenEnv × Scaler Hackathon, April 2026.

**Theme #3.1 — World Modeling / Professional Tasks**

---

## What problem does this solve?

Knowledge workers switch between apps hundreds of times a day — Gmail, Slack, Salesforce, Asana, Google Calendar. AI agents are expected to handle this kind of cross-app, long-horizon work. But training them requires an environment that grades **real outcomes**, not just text output.

Did the deal stage actually change in CRM? Did the meeting actually appear on the calendar? Did the reply go to the right person?

This environment provides exactly that infrastructure.

---

## What we built

An OpenEnv-compliant RL environment with:

- **5 simulated SaaS apps** — Email, Chat, CRM, Tasks, Calendar
- **20 tools** the agent can call (read_inbox, update_deal_stage, book_meeting, assign_task, etc.)
- **4 scenarios** from easy curriculum (2 tools) to full long-horizon coordination (8 tools)
- **4 independent reward functions** — tool correctness, efficiency, task completion, format validity
- **State-based success checks** — the environment verifies the world actually changed, not just that tools were called
- **Real Google Calendar integration** — the agent can book actual meetings via OAuth, not just simulated ones

---

## The 4 scenarios

| Scenario | Tools required | Description |
|---|---|---|
| ☀️ Morning Check-in | 2 | Curriculum scenario — read inbox + list channels |
| 🔥 Deal Rescue | 8 | Client cancels $50K contract — agent must save the deal |
| ⚔️ Team Conflict | 6 | Two engineers claimed the same task — agent resolves it |
| 🚀 Client Onboarding | 7 | New lead emails in — agent creates contact, tasks, meeting |

---

## Self-improvement demonstration

We demonstrated prompt-level self-improvement without any model retraining:

| Scenario | Run 1 reward | Run 2 reward (with memory) |
|---|---|---|
| Team Conflict | 1 | **21** |

After each episode, the agent extracts a lesson from its trajectory. The next run of the same scenario receives that lesson in its system prompt. Reward jumped from 1 to 21 — a 20-point improvement in one cycle.

---

## Reward design

Per the hackathon build guide — 4 independent reward functions instead of one scalar:

- **tool_correctness** — +1 per required tool called
- **tool_efficiency** — penalty for duplicates and unnecessary calls
- **task_completion** — +10 if state-based check passes
- **format_validity** — +2 if no malformed tool calls

TRL's GRPOTrainer receives `reward_funcs=[fn1, fn2, fn3, fn4]` and tracks each axis separately in metrics.

---

## Training pipeline

We ran 5 steps of GRPO training against this live Space using Qwen2.5-1.5B-Instruct on a T4 GPU:

- The pipeline completed end-to-end (289 seconds)
- The untrained 1.5B model does not yet emit parseable tool calls for Deal Rescue
- All 4 reward axes correctly return 0 for non-solutions (the reward function is working)
- Full training with SFT warmup + curriculum will be run onsite April 25-26 with HF compute credits

---

## Repository structure

```
scalar-x-meta-/                          ← project root (branch: hackathon-polish)
│
├── 📁 openenv/                          ← OpenEnv environment (the main deliverable)
│   └── enterprise_workflow_env/
│       ├── __init__.py                  ← package exports
│       ├── client.py                    ← HTTP/WebSocket client
│       ├── models.py                    ← EnterpriseAction + EnterpriseObservation
│       ├── openenv.yaml                 ← OpenEnv manifest
│       ├── pyproject.toml               ← Python package metadata
│       ├── uv.lock                      ← pinned dependency lock file
│       └── server/
│           ├── app.py                   ← FastAPI entry point
│           ├── Dockerfile               ← Docker image for HF Spaces
│           ├── requirements.txt         ← server dependencies
│           ├── enterprise_workflow_env_environment.py  ← core env (reset, step, 20 tools, rewards)
│           ├── mock_apps.py             ← 5 simulated SaaS apps (Docker copy)
│           ├── scenarios.py             ← 4 scenarios + success checks (Docker copy)
│           └── reward.py               ← episode scoring (Docker copy)
│
├── 📁 training/                         ← TRL training pipeline
│   ├── rollout.py                       ← runs one full episode over WebSocket
│   ├── train_grpo.py                    ← GRPO training script (TRL + Qwen2.5-1.5B)
│   ├── reward_curve.png                 ← reward curve from first training run
│   ├── trainer_state.json               ← TRL metrics (loss, rewards, step_time)
│   └── training_run_summary.json        ← human-readable training summary
│
├── 📁 integrations/                     ← real-world API integrations
│   ├── google_calendar.py               ← books actual meetings via Google Calendar v3 API
│   └── test_calendar_auth.py            ← OAuth verification script
│
├── agent.py                             ← Groq-powered agent (Llama-3.3-70B)
├── tools.py                             ← 20 tool schemas in OpenAI function-calling format
├── mock_apps.py                         ← 5 simulated SaaS apps (EmailApp, ChatApp, CRMApp, TaskApp, CalendarApp)
├── scenarios.py                         ← 4 scenarios + state-based success checks
├── reward.py                            ← single-scalar reward (used by dashboard)
├── reward_funcs.py                      ← 4 independent reward functions for GRPO
├── memory.py                            ← between-episode lesson storage
├── lesson_extractor.py                  ← post-episode lesson extraction via LLM
├── dashboard.py                         ← Streamlit live demo dashboard
├── REWARD_HACKING_AUDIT.md              ← exploit analysis for each reward axis
├── requirements.txt                     ← pinned project dependencies
└── .env.example                         ← template for GROQ_API_KEY setup
```

---

## Try it

**Interactive Playground** — use the Reset / Step / Get state buttons on the right to drive episodes manually.

**Python client:**

```python
from enterprise_workflow_env.client import EnterpriseWorkflowEnv
from enterprise_workflow_env.models import EnterpriseAction

with EnterpriseWorkflowEnv(
    base_url="https://yashwanthprabhu-enterprise-workflow-env.hf.space"
).sync() as env:
    result = env.reset()
    obs = result.observation
    print("Scenario:", obs.scenario_id)
    print("Task:", obs.tool_result[:200])
```

---

## Links

- **GitHub repo:** https://github.com/yashwanthprabhu07/scalar-x-meta- (branch: `hackathon-polish`)
- **Training artifacts:** [reward_curve.png](https://github.com/yashwanthprabhu07/scalar-x-meta-/blob/hackathon-polish/training/reward_curve.png)
- **Demo video:** coming soon
- **Team:** Yashwanth Prabhu R, Vivek Gowda NV, Chaitanya S Shetty

---

*Built for the Meta PyTorch OpenEnv × Scaler School of Technology Hackathon, Bangalore, April 25-26, 2026.*