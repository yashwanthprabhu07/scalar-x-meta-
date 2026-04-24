# ============================================================
# training/train_grpo.py — GRPO training with multiple reward functions
#
# Updated to use 4 independent reward functions instead of 1 scalar,
# per the hackathon build guide recommendation (Section 7).
#
# ── How to use ───────────────────────────────────────────────
# Local smoke test (no training, just validate pipeline):
#   python training/train_grpo.py --dry-run
#
# Real training (requires CUDA GPU):
#   python training/train_grpo.py
#
# In Colab: paste CELL N blocks as separate cells.
# ============================================================

# ═══════════════════════════════════════════════════════════
# CELL 1 — INSTALL DEPS (uncomment in Colab)
# ═══════════════════════════════════════════════════════════
# !pip install -q openenv-core trl transformers peft accelerate
# !pip install -q unsloth  # optional — faster training

# ═══════════════════════════════════════════════════════════
# CELL 2 — IMPORTS AND CONFIG
# ═══════════════════════════════════════════════════════════
import json
import os
import sys
from typing import Any, Dict, List, Optional

# ── Environment config ──
ENV_SPACE_URL = "https://yashwanthprabhu-enterprise-workflow-env.hf.space"

# ── Model / training config ──
MODEL_NAME        = "Qwen/Qwen2.5-1.5B-Instruct"
NUM_TRAIN_STEPS   = 5
NUM_GENERATIONS   = 4
MAX_EPISODE_STEPS = 15
LEARNING_RATE     = 1e-5
BATCH_SIZE        = 1


# ═══════════════════════════════════════════════════════════
# CELL 3 — IMPORT THE ROLLOUT HELPER
# ═══════════════════════════════════════════════════════════
# In Colab, uncomment these:
# !git clone https://github.com/yashwanthprabhu07/scalar-x-meta-.git
# %cd scalar-x-meta-
# !git checkout hackathon-polish
# sys.path.insert(0, ".")
# sys.path.insert(0, "openenv")
# sys.path.insert(0, "training")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
_PROJECT  = os.path.dirname(_THIS_DIR) if _THIS_DIR.endswith("training") else _THIS_DIR
for p in (_PROJECT, os.path.join(_PROJECT, "openenv"), os.path.join(_PROJECT, "training")):
    if p not in sys.path:
        sys.path.insert(0, p)

from rollout import run_episode, AgentFn


# ═══════════════════════════════════════════════════════════
# CELL 4 — BASELINE / SCRIPTED AGENT FOR TESTING
# ═══════════════════════════════════════════════════════════
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
def load_model_and_tokenizer(model_name: str):
    """Load model with Unsloth for faster training and lower VRAM usage."""
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=1024,
            dtype=None,           # auto-detect: bfloat16 on Ampere+, float16 otherwise
            load_in_4bit=True,    # QLoRA — halves VRAM usage vs full precision
        )
        # Add LoRA adapters for efficient fine-tuning
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,                 # LoRA rank
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        print("   Using Unsloth (4-bit QLoRA)")
    except Exception as e:
        # Fallback to standard transformers (CPU or no Unsloth)
        print(f"   Unsloth not available ({e}), using standard transformers")
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
    AgentFn that uses the HF model's chat template (with tools) to pick
    the next tool call. Uses Qwen's built-in tool-calling format.
    """
    import torch

    def _agent(system_prompt, conversation, tool_schemas):
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation)

        inputs = tokenizer.apply_chat_template(
            messages,
            tools=tool_schemas,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

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

        return _parse_qwen_tool_call(response)

    return _agent


def _parse_qwen_tool_call(response: str) -> Optional[Dict]:
    """Extract a tool call from Qwen's response string."""
    import re
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", response, re.DOTALL)
    if not match:
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
# CELL 6 — MULTI-FUNCTION REWARDS (build-guide aligned)
# ═══════════════════════════════════════════════════════════
# Per the hackathon build guide (Section 7):
#   "use multiple independent reward functions, not just one.
#    If you only have a single reward signal, it is easier
#    for the model to hack it."
#
# TRL's GRPOTrainer accepts `reward_funcs=[fn1, fn2, ...]` and
# tracks each separately in metrics. Each reward function follows
# the same signature and extracts one axis of the trajectory score.

def _run_rollouts_for_prompts(prompts, model, tokenizer, env_url: str) -> List[Dict]:
    """Run one rollout per prompt. Returns trajectory dicts."""
    agent = make_model_agent(model, tokenizer)
    trajectories = []
    for _ in range(len(prompts)):
        try:
            result = run_episode(
                agent_fn=agent,
                base_url=env_url,
                max_steps=MAX_EPISODE_STEPS,
                verbose=False,
            )
            trajectories.append(result)
        except Exception as e:
            print(f"   ⚠️  rollout failed: {e}")
            trajectories.append({
                "required_actions": [],
                "taken_actions": [],
                "task_success": False,
                "steps": [],
                "final_reward": -10.0,
            })
    return trajectories


def make_multi_reward_fns(model, tokenizer, env_url: str = ENV_SPACE_URL):
    """
    Returns a list of 4 reward functions for TRL's GRPOTrainer.

    Each function re-runs rollouts per prompt (TRL calls each reward
    function separately). For our small training (5 steps × 4 gens),
    the cost is acceptable. Production code would cache.
    """
    from reward_funcs import (
        tool_correctness_reward,
        tool_efficiency_reward,
        task_completion_reward,
        format_validity_reward,
    )

    def _make_axis_fn(axis_name: str, score_fn):
        def _reward_fn(prompts, completions, **kwargs):
            trajectories = _run_rollouts_for_prompts(prompts, model, tokenizer, env_url)
            return [float(score_fn(t)) for t in trajectories]
        _reward_fn.__name__ = f"reward_{axis_name}"
        return _reward_fn

    return [
        _make_axis_fn("tool_correctness", tool_correctness_reward),
        _make_axis_fn("tool_efficiency",  tool_efficiency_reward),
        _make_axis_fn("task_completion",  task_completion_reward),
        _make_axis_fn("format_validity",  format_validity_reward),
    ]


# ═══════════════════════════════════════════════════════════
# CELL 7 — EVAL LOOP
# ═══════════════════════════════════════════════════════════
def eval_agent(agent_fn: AgentFn, env_url: str, num_episodes: int = 3,
               label: str = "agent", verbose: bool = True) -> Dict:
    """Run `num_episodes` episodes; report mean reward + success rate."""
    from reward_funcs import score_trajectory

    results = []
    for i in range(num_episodes):
        result = run_episode(
            agent_fn=agent_fn,
            base_url=env_url,
            max_steps=MAX_EPISODE_STEPS,
            verbose=False,
        )
        scored = score_trajectory(result)
        results.append(scored)
        if verbose:
            print(f"   [{label} ep {i+1}] total={scored['total']:+.1f}  "
                  f"(correct={scored['tool_correctness']:+.0f}, "
                  f"eff={scored['tool_efficiency']:+.1f}, "
                  f"done={scored['task_completion']:+.0f}, "
                  f"fmt={scored['format_validity']:+.0f})")

    mean_total = sum(r["total"] for r in results) / max(len(results), 1)
    mean_completion = sum(r["task_completion"] for r in results) / max(len(results), 1)
    print(f"── {label}: mean_total={mean_total:+.2f}, "
          f"mean_completion={mean_completion:+.2f}")
    return {
        "mean_total": mean_total,
        "mean_completion": mean_completion,
        "per_episode": results,
    }


# ═══════════════════════════════════════════════════════════
# CELL 8 — GRPO TRAINING LOOP
# ═══════════════════════════════════════════════════════════
def train_grpo():
    """Run a short GRPO training run with 4 independent reward funcs."""
    from trl import GRPOConfig, GRPOTrainer

    print("── Loading model:", MODEL_NAME)
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n── Baseline evaluation (pre-training) ──")
    baseline_agent = make_model_agent(model, tokenizer)
    baseline_stats = eval_agent(baseline_agent, ENV_SPACE_URL, num_episodes=2, label="baseline")

    scenario_prompt = (
        "You are an AI sales assistant. Acme Corp emailed to cancel their $50,000 "
        "contract. Use the available tools to read their email, look up the deal, "
        "reply with a 10% discount offer, update the deal stage to Negotiation, "
        "add a note, book a follow-up meeting, and post to the sales channel."
    )
    dataset = [{"prompt": scenario_prompt} for _ in range(NUM_TRAIN_STEPS * BATCH_SIZE)]

    reward_fns = make_multi_reward_fns(model, tokenizer, env_url=ENV_SPACE_URL)
    print(f"\n── Using {len(reward_fns)} independent reward functions ──")
    for fn in reward_fns:
        print(f"   • {fn.__name__}")

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
        save_steps=NUM_TRAIN_STEPS,
        gradient_accumulation_steps=1,
        bf16=True,
    )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        reward_funcs=reward_fns,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"\n── Starting GRPO training ({NUM_TRAIN_STEPS} steps, {NUM_GENERATIONS} gens/step) ──")
    trainer.train()
    print("── Training complete ──")

    print("\n── Post-training evaluation ──")
    trained_agent = make_model_agent(model, tokenizer)
    trained_stats = eval_agent(trained_agent, ENV_SPACE_URL, num_episodes=2, label="trained")

    print("\n══════════════════════════════════════")
    print(f"BASELINE:  mean_total={baseline_stats['mean_total']:+.2f}  completion={baseline_stats['mean_completion']:+.2f}")
    print(f"TRAINED:   mean_total={trained_stats['mean_total']:+.2f}  completion={trained_stats['mean_completion']:+.2f}")
    print(f"Δ total:   {trained_stats['mean_total'] - baseline_stats['mean_total']:+.2f}")
    print("══════════════════════════════════════")

    return baseline_stats, trained_stats


# ═══════════════════════════════════════════════════════════
# CELL 9 — DRY RUN (no GPU needed)
# ═══════════════════════════════════════════════════════════
def dry_run():
    """Validate the pipeline without training. Uses scripted agent."""
    print("Dry run: scripted agent → live Space, multi-reward scoring\n")

    from reward_funcs import score_trajectory

    agent = make_deal_rescue_scripted_agent()
    result = run_episode(agent_fn=agent, base_url=ENV_SPACE_URL,
                         max_steps=15, verbose=True)

    scored = score_trajectory(result)
    print("\n── Multi-reward scoring ──")
    for line in scored["breakdown"]:
        print(f"  {line}")
    print(f"  {'═' * 40}")
    print(f"  Total:              {scored['total']:+.1f}")
    print(f"\nEnv final_reward:   {result['final_reward']}")
    print(f"Task success:       {result['task_success']}")


# ═══════════════════════════════════════════════════════════
# CELL 10 — ENTRY POINT
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
    else:
        train_grpo()