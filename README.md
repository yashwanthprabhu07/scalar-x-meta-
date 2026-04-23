# AI Enterprise Workflow Environment

> An **OpenEnv-compliant environment** for training LLM agents on long-horizon, multi-app enterprise workflows. Built for the **Meta PyTorch OpenEnv × Scaler School of Technology Hackathon** (April 2026).

**Theme #3.1 — World Modeling / Professional Tasks**

**[Try the live environment →](https://huggingface.co/spaces/yashwanthprabhu/enterprise-workflow-env)**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenEnv 0.2.3](https://img.shields.io/badge/OpenEnv-0.2.3-green.svg)](https://github.com/meta-pytorch/OpenEnv)
[![TRL](https://img.shields.io/badge/TRL-GRPO-purple.svg)](https://huggingface.co/docs/trl)
[![HF Spaces](https://img.shields.io/badge/deployed-HF%20Spaces-yellow.svg)](https://huggingface.co/spaces/yashwanthprabhu/enterprise-workflow-env)

---

## The problem

Modern AI agents are increasingly expected to handle **cross-app enterprise workflows** — the kind a human employee does every day: read an email, check a CRM deal, reply with a tailored offer, book a meeting, post a Slack update. These workflows require long-horizon planning, real-world state awareness, and integration with multiple SaaS tools.

Training agents for this is hard. You cannot grade text output alone — you must grade **outcomes**: did the deal actually get updated in CRM? Did the meeting actually appear on the calendar? Did the reply go to the right person?

This project builds the infrastructure to train and evaluate such agents:

1. **An OpenEnv-compliant environment** modeling 5 mock SaaS apps with 20 tools.
2. **4 scenarios** ranging from easy curriculum tasks to 8-tool long-horizon coordination.
3. **4 independent reward functions** — hack-resistant by design.
4. **A real Google Calendar integration** — the agent can book actual meetings, not just simulated ones.
5. **A complete TRL training pipeline** ready for GRPO on the live environment.

---

## Live environment

- **Space page:** https://huggingface.co/spaces/yashwanthprabhu/enterprise-workflow-env
- **Server endpoint:** https://yashwanthprabhu-enterprise-workflow-env.hf.space
- **Interactive Playground:** click Reset / Step / Get state in the Space UI to drive episodes by hand.

Python client example:

```python
from enterprise_workflow_env.client import EnterpriseWorkflowEnv
from enterprise_workflow_env.models  import EnterpriseAction

with EnterpriseWorkflowEnv(
    base_url="https://yashwanthprabhu-enterprise-workflow-env.hf.space"
).sync() as env:
    result = env.reset()
    result = env.step(EnterpriseAction(
        tool_name="read_inbox", tool_args={},
    ))
```

---

## Environment design

### The 5 mock apps (20 tools)

| App | Tools |
|---|---|
| **Email** | read_inbox, read_email, send_email, reply_email |
| **Chat** | list_channels, read_channel, post_message |
| **CRM** | get_deal, update_deal_stage, add_note, get_contact, create_contact |
| **Tasks** | list_tasks, get_task, create_task, assign_task, close_task |
| **Calendar** | list_meetings, check_conflicts, book_meeting |

### The 4 scenarios

| Scenario | Required tools | Purpose |
|---|---|---|
| Morning Check-in | 2 | Easy curriculum — gives untrained models a non-zero reward signal (build-guide Section 6) |
| Deal Rescue | 8 | Acme Corp cancels their contract. Agent must read, offer discount, update deal, book follow-up, post to sales channel |
| Team Conflict | 6 | Two engineers claimed the same task. Agent must reassign, resolve, and schedule a sync |
| Client Onboarding | 7 | New lead emails expressing interest. Agent must create contact, onboarding tasks, kickoff meeting, welcome reply |

### Sparse reward, long-horizon

Per build-guide Section 2: 0 reward per intermediate step, full episode reward on terminal step.
reset()   -> scenario context + agent_prompt + required_actions
step()    -> tool_result + sparse_reward (0)
step()    -> tool_result + sparse_reward (0)
...
step()    -> full_episode_reward + done=True
(fires when required tools called AND state_check passes)

---

## Reward design (hack-resistant)

Per build-guide **Section 7** — 4 independent reward functions instead of 1 scalar:

| Function | Range | What it rewards |
|---|---|---|
| tool_correctness_reward | 0 to N | Calling each required tool at least once |
| tool_efficiency_reward | -K to 0 | Penalizes duplicates (-0.5) and unnecessary calls (-1.0) |
| task_completion_reward | 0 or +10 | State-based check: did the apps actually end up in the right state? |
| format_validity_reward | 0 or +2 | No errors, no hallucinated tools, no missing args |

TRL's GRPOTrainer takes `reward_funcs=[fn1, fn2, fn3, fn4]` as a list and tracks each axis independently in metrics.

### State-based success checks

Every scenario defines a `success_check(apps) -> (passed, reasons)` function that inspects the final state of the mock apps. For Deal Rescue this checks:

- Acme deal stage updated to Negotiation (not just that `update_deal_stage` was called — actual state changed)
- A new note added to the Acme deal
- A follow-up meeting exists (not in the seeded set)
- A reply was sent to the Acme contact
- A message was posted in the sales channel by the agent

An agent calling `book_meeting` with garbage args passes the tool-call check but FAILS the state check.

See [REWARD_HACKING_AUDIT.md](REWARD_HACKING_AUDIT.md) for a full exploit analysis.

---

## Real-world Google Calendar integration

Most hackathon projects stop at simulation. This one goes further — **the same agent code can book real meetings on a real Google Calendar** via OAuth.

```python
from integrations.google_calendar import RealCalendarApp

cal = RealCalendarApp()
cal.book_meeting(
    title="Hackathon Demo Meeting",
    attendees=["rajesh.kumar@acmecorp.com"],
    date="2026-04-26", time="14:00", duration_mins=30,
)
# -> Creates a real event on your Google Calendar
```

Uses Google Calendar API v3 with OAuth 2.0 installed-app flow. Scopes limited to `/auth/calendar`. Credentials never committed to git.

---

## Self-improvement demonstration

Before-and-after evidence that the agent learns without any model retraining:

| Scenario | Run 1 reward | Run 2 reward (with memory loop) | Delta |
|---|---|---|---|
| Team Conflict | 1 | 21 | **+20** |

After each episode, an LLM extracts a 1-2 sentence lesson from the trajectory and final reward. The next episode of the same scenario receives these lessons in its system prompt. No weights changed — but behavior changes.

The reward signal the memory loop uses is the **same one a PyTorch policy would use**. When plugged into OpenEnv + GRPO (see training below), the reward function works unchanged.

---

## Training pipeline

Minimal GRPO training script at `training/train_grpo.py`:

```bash
# Dry-run (no GPU, scripted agent validates pipeline)
python training/train_grpo.py --dry-run

# Full training (requires CUDA)
python training/train_grpo.py
```

- Built on HuggingFace TRL (GRPOTrainer)
- Passes the 4 independent reward functions to `reward_funcs=[...]`
- Connects to the live HF Space over WebSocket
- Target model: Qwen2.5-1.5B-Instruct (fits on T4 free tier)

Validated end-to-end via dry-run: scripted agent -> live HF Space -> multi-reward scoring -> total: +20.0, task_success: True.

---

## Quickstart

```bash
git clone https://github.com/yashwanthprabhu07/scalar-x-meta-.git
cd scalar-x-meta-
git checkout hackathon-polish
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                # Add your GROQ_API_KEY
streamlit run dashboard.py          # Opens at localhost:8501
```

For real Google Calendar integration, see `integrations/test_calendar_auth.py` for OAuth setup.

---

## Project structure
scalar-x-meta-/
|-- openenv/enterprise_workflow_env/      # OpenEnv environment (deployed)
|   |-- client.py
|   |-- models.py
|   |-- openenv.yaml
|   -- server/
|       |-- app.py
|       |-- enterprise_workflow_env_environment.py
|       |-- mock_apps.py
|       |-- scenarios.py
|       -- reward.py
|-- training/
|   |-- rollout.py                         # Episode runner (WebSocket)
|   -- train_grpo.py                      # GRPO trainer (TRL)
|-- integrations/
|   |-- google_calendar.py                 # Real Calendar API wrapper
|   -- test_calendar_auth.py              # OAuth verification
|-- agent.py                               # Baseline Groq-powered agent
|-- tools.py                               # 20 tool schemas (OpenAI format)
|-- mock_apps.py                           # Dashboard copy
|-- scenarios.py                           # Dashboard copy (4 scenarios)
|-- reward_funcs.py                        # 4 independent reward fns
|-- memory.py                              # Between-episode lessons
|-- dashboard.py                           # Streamlit demo UI
-- REWARD_HACKING_AUDIT.md

---

## Tech stack

- **OpenEnv 0.2.3** — environment protocol
- **HuggingFace TRL** — GRPO trainer
- **HuggingFace Spaces** — environment hosting
- **FastAPI** — HTTP server
- **Google Calendar API v3** — real-world integration
- **Groq + Llama-3.3-70B** — baseline agent LLM
- **Streamlit** — live demo dashboard
- **Qwen2.5-1.5B-Instruct** — training target

---

## How this addresses the hackathon themes

**Primary: Theme #3.1 — World Modeling / Professional Tasks.**
Captures nuances of a partially observable enterprise world across 5 apps. Real Calendar integration extends to actual SaaS APIs.

**Also:**
- **Theme #2 (Long-Horizon):** Deal Rescue = 8 coordinated calls, sparse delayed reward.
- **Theme #4 (Self-Improvement):** memory loop: Team Conflict 1 -> 21 without retraining.

---

## Submission materials

- **Live Space:** https://huggingface.co/spaces/yashwanthprabhu/enterprise-workflow-env
- **GitHub repo:** https://github.com/yashwanthprabhu07/scalar-x-meta- (branch: `hackathon-polish`)
- **Mini-blog:** _coming soon — will be posted on HuggingFace_
- **Demo video:** _coming soon — will be on YouTube_
- **Training reward plots:** _coming soon — from live GRPO run_

---

## Team

**Yashwanth Prabhu R** (team lead), Vivek Gowda NV, Chaitanya S Shetty.

Submitted for the Meta PyTorch OpenEnv × Scaler School of Technology Hackathon, Bangalore, April 25-26, 2026.

---

## Acknowledgments

- **Meta PyTorch** for OpenEnv
- **HuggingFace** for TRL, Spaces hosting, and models
- **Scaler School of Technology** for organizing the hackathon and providing compute credits

---

## License

MIT.
