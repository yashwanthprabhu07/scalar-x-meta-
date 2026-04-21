# ============================================================
# training/train_grpo.py — GRPO training against the OpenEnv Space
#
# This script (or Colab notebook) fine-tunes a small LLM using
# GRPO (Group Relative Policy Optimization) against our live
# Enterprise Workflow OpenEnv environment hosted on HF Spaces.
#
# What it demonstrates:
#   - End-to-end RL training pipeline on a real agentic task
#   - Reward shaping from the OpenEnv reward function
#   - A genuine reward curve showing improvement
#
# Designed for Colab free tier (T4 GPU, 16GB VRAM) but runs
# anywhere with a CUDA GPU.
#
# ── How to use ───────────────────────────────────────────────
# Local test (no training, just validate pipeline):
#   python training/train_grpo.py --dry-run
#
# Local training (requires CUDA GPU):
#   python training/train_grpo.py
#
# In Colab:
#   paste each "# CELL N" block as a separate cell
# ============================================================

# ═══════════════════════════════════════════════════════════
# CELL 1 — INSTALL DEPS
# ═══════════════════════════════════════════════════════════
# (In Colab, run this as the first cell)
# !pip install -q openenv-core trl transformers peft accelerate
# !pip install -q unsloth  # optional — faster training, but optional

# ═══════════════════════════════════════════════════════════
# CELL 2 — IMPORTS AND CONFIG
# ═══════════════════════════════════════════════════════════
import json
import os
import sys
from typing import Any, Dict, List, Optional

# ── Environment config ──
# This is YOUR live Space URL. Change if you redeploy under a different name.
ENV_SPACE_URL = "https://yashwanthprabhu-enterprise-workflow-env.hf.space"

# ── Model config ──
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
# Training hyperparameters are kept small so this finishes on free-tier Colab
# in ~10-15 minutes. For real training, scale these up.
NUM_TRAIN_STEPS   = 5       # number of GRPO optimization steps
NUM_GENERATIONS   = 4       # rollouts per prompt (GRPO needs >= 2)
MAX_EPISODE_STEPS = 15      # safety cap on tool calls per rollout
LEARNING_RATE     = 1e-5
BATCH_SIZE        = 1       # 1 prompt per step × NUM_GENERATIONS rollouts each


# ═══════════════════════════════════════════════════════════
# CELL 3 — IMPORT THE ROLLOUT HELPER
# ═══════════════════════════════════════════════════════════
# In Colab we need to clone the repo first to get rollout.py.
# Uncomment these in Colab:
# !git clone https://github.com/yashwanthprabhu07/scalar-x-meta-.git
# %cd scalar-x-meta-
# !git checkout hackathon-polish
# sys.path.insert(0, ".")
# sys.path.insert(0, "openenv")
# sys.path.insert(0, "training")

# Locally we're already in the project — fix up paths so rollout.py works
_THIS_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
_PROJECT  = os.path.dirname(_THIS_DIR) if _THIS_DIR.endswith("training") else _THIS_DIR
for p in (_PROJECT, os.path.join(_PROJECT, "openenv"), os.path.join(_PROJECT, "training")):
    if p not in sys.path:
        sys.path.insert(0, p)

from rollout import run_episode, AgentFn


# ═══════════════════════════════════════════════════════════
# CELL 4 — BASELINE AGENT (GROQ or a scripted baseline)
# ═══════════════════════════════════════════════════════════
# Before training, we establish a baseline reward by running our
# existing Groq-powered agent a few times against the Space and
# averaging the rewards.
#
# For Colab, we swap this for the HF model's initial behavior.
# For local dry-run, we just use a scripted agent.
#
# Each function below implements the AgentFn signature:
#   (system_prompt, conversation, tool_schemas) -> {tool_name, tool_args} | None

def make_deal_rescue_scripted_agent() -> AgentFn:
    """Scripted perfect-play agent for smoke testing."""
    script = [
        {"tool_name": "read_inbox",        "tool_args": {}},
        {"tool_name": "read_email",        "tool_args": {"email_id": "email_001"}},
        {"tool_name": "get_deal",          "tool_args": {"deal_id": "deal_001"}},
        {"tool_name": "reply_email",       "tool_args": {
            "email_id": "email_001",
            "body": "We value your business. Here is a 10% discount to keep you."
        }},
        {"tool_name": "update_deal_stage", "tool_args": {
            "deal_id": "deal_001", "new_stage": "Negotiation"
        }},
        {"tool_name": "add_note",          "tool_args": {
            "deal_id": "deal_001", "note": "Offered 10% discount."
        }},
        {"tool_name": "book_meeting",      "tool_args": {
            "title": "Acme Follow-up",
            "attendees": ["rajesh.kumar@acmecorp.com"],
            "date": "2026-04-23", "time": "10:00", "duration_mins": 30,
        }},
        {"tool_name": "post_message",      "tool_args": {
            "channel": "sales",
            "message": "Acme Corp offered 10% discount, follow-up booked."
        }},
    ]
    remaining = list(script)
    def _agent(system_prompt, conversation, tool_schemas):
        if not remaining:
            return None
        return remaining.pop(0)
    return _agent


# ═══════════════════════════════════════════════════════════
# CELL 5 — MODEL-BACKED AGENT
# ═══════════════════════════════════════════════════════════
# This is the agent that TRL will be training. It takes the current
# conversation and uses the model's chat template (which supports
# tool calling) to pick the next action.
#
# In Colab this loads Qwen2.5-1.5B. Locally it's skipped unless we
# have a GPU (it's a 3GB download).

def load_model_and_tokenizer(model_name: str):
    """Lazy import so local dry-runs don't require transformers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, tokenizer


def make_model_agent(model, tokenizer, max_new_tokens: int = 200) -> AgentFn:
    """
    Returns an AgentFn that uses the given HF model to pick the next tool call.
    Uses Qwen's built-in chat-template tool-calling format.
    """
    import torch

    def _agent(system_prompt, conversation, tool_schemas):
        # Build messages in the format Qwen expects
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation)

        # Apply the chat template with tools
        inputs = tokenizer.apply_chat_template(
            messages,
            tools=tool_schemas,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        # Generate a completion
        with torch.no_grad():
            output_ids = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=False)

        # Extract the tool call from the response
        # Qwen emits: <tool_call>\n{"name": "x", "arguments": {...}}\n</tool_call>
        parsed = _parse_qwen_tool_call(response)
        return parsed

    return _agent


def _parse_qwen_tool_call(response: str) -> Optional[Dict]:
    """Extract a tool call from Qwen's response string."""
    import re
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", response, re.DOTALL)
    if not match:
        # Model didn't emit a tool call — treat as 'done'
        return None
    try:
        parsed = json.loads(match.group(1))
        return {
            "tool_name": parsed.get("name") or parsed.get("tool_name"),
            "tool_args": parsed.get("arguments") or parsed.get("tool_args") or {},
        }
    except json.JSONDecodeError:
        return None


# ═══════════════════════════════════════════════════════════
# CELL 6 — REWARD FUNCTION FOR TRL
# ═══════════════════════════════════════════════════════════
# TRL's GRPOTrainer expects a reward function of signature:
#   def reward_fn(prompts, completions, **kwargs) -> List[float]
#
# For OpenEnv-based training, the "completion" isn't a single string —
# it's a whole trajectory of tool calls. So we need to:
#   1. Have the model roll out a trajectory against the env for each prompt
#   2. Return the final episode reward from the env
#
# (The model object is captured in the closure.)

def make_openenv_reward_fn(model, tokenizer, env_url: str = ENV_SPACE_URL):
    """
    Factory: returns a callable suitable for TRL's GRPOTrainer.

    When called, it runs one episode per prompt against the live
    OpenEnv server and returns the final reward.
    """
    def _reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        rewards = []
        agent = make_model_agent(model, tokenizer)
        for _ in range(len(prompts)):
            try:
                result = run_episode(
                    agent_fn=agent,
                    base_url=env_url,
                    max_steps=MAX_EPISODE_STEPS,
                    verbose=False,
                )
                rewards.append(float(result["final_reward"]))
            except Exception as e:
                print(f"   ⚠️  rollout failed: {e}")
                rewards.append(-10.0)  # failure sentinel
        return rewards

    return _reward_fn


# ═══════════════════════════════════════════════════════════
# CELL 7 — EVAL LOOP (runs N episodes, reports stats)
# ═══════════════════════════════════════════════════════════
def eval_agent(agent_fn: AgentFn, env_url: str, num_episodes: int = 3,
               label: str = "agent", verbose: bool = True) -> Dict:
    """Run `num_episodes` episodes; return mean reward + success rate."""
    rewards = []
    successes = []
    for i in range(num_episodes):
        # Re-create a fresh scripted-copy if needed; for model-backed it's stateless
        result = run_episode(
            agent_fn=agent_fn if not hasattr(agent_fn, "_needs_reset") else agent_fn(),
            base_url=env_url,
            max_steps=MAX_EPISODE_STEPS,
            verbose=False,
        )
        rewards.append(result["final_reward"])
        successes.append(bool(result.get("task_success")))
        if verbose:
            print(f"   [{label} ep {i+1}] reward={result['final_reward']:.1f}, success={result.get('task_success')}")

    mean_reward = sum(rewards) / max(len(rewards), 1)
    success_rate = sum(successes) / max(len(successes), 1)
    print(f"── {label}: mean_reward={mean_reward:.2f}, success_rate={success_rate:.2%}")
    return {
        "mean_reward": mean_reward,
        "success_rate": success_rate,
        "rewards": rewards,
    }


# ═══════════════════════════════════════════════════════════
# CELL 8 — GRPO TRAINING LOOP (the main event)
# ═══════════════════════════════════════════════════════════
def train_grpo():
    """Run a short GRPO training run against the live OpenEnv Space."""
    from trl import GRPOConfig, GRPOTrainer

    print("── Loading model:", MODEL_NAME)
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    print(f"   Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # BASELINE — before training, see how the model does cold
    print("\n── Baseline evaluation (pre-training) ──")
    baseline_agent = make_model_agent(model, tokenizer)
    baseline_stats = eval_agent(baseline_agent, ENV_SPACE_URL, num_episodes=2, label="baseline")

    # Build the dataset — just repeated copies of the scenario prompt.
    # TRL will sample NUM_GENERATIONS rollouts per prompt.
    scenario_prompt = (
        "You are an AI sales assistant. Acme Corp emailed to cancel their $50,000 "
        "contract. Use the available tools to read their email, look up the deal, "
        "reply with a 10% discount offer, update the deal stage to Negotiation, "
        "add a note, book a follow-up meeting, and post to the sales channel."
    )
    dataset = [{"prompt": scenario_prompt} for _ in range(NUM_TRAIN_STEPS * BATCH_SIZE)]

    reward_fn = make_openenv_reward_fn(model, tokenizer, env_url=ENV_SPACE_URL)

    # GRPO config — tiny run for demo
    config = GRPOConfig(
        output_dir="./grpo_output",
        per_device_train_batch_size=BATCH_SIZE,
        num_generations=NUM_GENERATIONS,
        max_prompt_length=1024,
        max_completion_length=400,
        learning_rate=LEARNING_RATE,
        num_train_epochs=1,
        max_steps=NUM_TRAIN_STEPS,
        logging_steps=1,
        save_steps=NUM_TRAIN_STEPS,  # save only at end
        gradient_accumulation_steps=1,
        bf16=True,
    )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        reward_funcs=[reward_fn],
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"\n── Starting GRPO training ({NUM_TRAIN_STEPS} steps, {NUM_GENERATIONS} generations/step) ──")
    trainer.train()
    print("── Training complete ──")

    # POST-TRAINING EVALUATION
    print("\n── Post-training evaluation ──")
    trained_agent = make_model_agent(model, tokenizer)
    trained_stats = eval_agent(trained_agent, ENV_SPACE_URL, num_episodes=2, label="trained")

    print("\n══════════════════════════════════════")
    print(f"BASELINE:   mean_reward={baseline_stats['mean_reward']:.2f}, success_rate={baseline_stats['success_rate']:.2%}")
    print(f"TRAINED:    mean_reward={trained_stats['mean_reward']:.2f}, success_rate={trained_stats['success_rate']:.2%}")
    print(f"Δ reward:   {trained_stats['mean_reward'] - baseline_stats['mean_reward']:+.2f}")
    print("══════════════════════════════════════")

    return baseline_stats, trained_stats


# ═══════════════════════════════════════════════════════════
# CELL 9 — DRY RUN (for local testing; no GPU needed)
# ═══════════════════════════════════════════════════════════
def dry_run():
    """
    Validate the pipeline without training.
    Runs a scripted agent against the live Space to confirm reachability.
    """
    print("Dry run: scripted agent → live Space")
    agent = make_deal_rescue_scripted_agent()
    result = run_episode(agent_fn=agent, base_url=ENV_SPACE_URL,
                         max_steps=15, verbose=True)
    print(f"\n→ Final reward: {result['final_reward']}")
    print(f"→ Task success: {result['task_success']}")


# ═══════════════════════════════════════════════════════════
# CELL 10 — ENTRY POINT
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Run pipeline smoke test without training")
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
    else:
        train_grpo()