# SQL Reasoning LLM 

> **A note on AI use:** this is a learning project, and I build it with AI models — Gemini, Claude Sonnet and Claude Opus — as collaborators.

An end to end machine learning pipeline that trains a compact, 0.5B parameter language model to autonomously convert natural language questions into highly accurate SQL queries. 

Unlike standard instruct-tuning, this project builds a **Reasoning Engine** from scratch using a 3-stage pipeline: Continual Pre-Training (CPT), Supervised Fine-Tuning (SFT), and Reinforcement Learning via Group Relative Policy Optimization (GRPO). The model learns to "think" before it speaks, grading its own logic using verifiable reward functions.

**Base Model:** `Qwen/Qwen2.5-0.5B`  
**Frameworks:** PyTorch, Hugging Face `trl`, Unsloth (for LoRA/4-bit optimization)

## Pipeline Architecture

1. **Phase 1: Continual Pre-Training (CPT)** - Ingests raw database schemas and SQL queries to teach the model the base domain vocabulary and syntax.
2. **Phase 2: Supervised Fine-Tuning (SFT)** - Uses ChatML formatting to enforce a strict XML style behavioral contract: `<think> [logic] </think> <answer> [sql] </answer>`.
3. **Phase 3: Reinforcement Learning (GRPO)** - The model generates multiple thought paths per prompt. Reward functions score the output based on strict formatting (+1.0) and SQL ground truth correctness (+2.0). The model updates its policy to favor high scoring logical reasoning.

## Core Features
- **Memory Efficient Training:** Utilizes 4 bit quantization and LoRA adapters via Unsloth, allowing the entire pipeline to run on a single 24GB consumer grade GPU.
- **Verifiable Rewards:** No separate critic model required. The GRPO loop uses deterministic regex and string matching to calculate advantage.
- **Modern Python Tooling:** Environment fully managed by Astral's `uv` package manager.

## Project Structure

```text
data/
	processed/
		cpt_training_data.jsonl
		sft_training_data.jsonl
models/
    All saved models are here
src/
	sqlrl/
		tokenizer.py
		data_prep/
			format_cpt_data.py
			format_sft_data.py
		training/
			model_loader.py
			cpt_trainer.py
			sft_trainer.py
			grpo_trainer.py
			merge.py
		serving/
			inference.py
			api.py
			agent.py
infra/
	Launch/stop/status scripts for the EC2 GPU boxes
```


## Requirements

- Linux + NVIDIA GPU (CUDA)
- Python `>=3.13` (as configured in `pyproject.toml`)
- `uv` package manager
- Hugging Face access token
- Wandb access token to log metrics(optional)

## Installation

```bash
uv sync
```

## Hugging Face Authentication

If `huggingface-cli` is not found, use:

```bash
uvx --from huggingface_hub hf auth login
```

Or non interactive:

```bash
export HF_TOKEN="your_token_here"
uvx --from huggingface_hub hf auth login --token "$HF_TOKEN"
```

## Data Preparation

Build both datasets from `b-mc2/sql-create-context`:

```bash
uv run python -m sqlrl.data_prep.format_cpt_data
uv run python -m sqlrl.data_prep.format_sft_data
```

Outputs:

- `data/processed/cpt_training_data.jsonl`
- `data/processed/sft_training_data.jsonl`

## Training Pipeline

Run stages in order.

### 1) CPT

```bash
uv run python -m sqlrl.training.cpt_trainer
```

Saves adapters to:

- `models/qwen-0.5b-cpt-lora`

### 2) SFT

```bash
uv run python -m sqlrl.training.sft_trainer
```

Saves adapters to:

- `models/qwen-0.5b-sft-lora`

### 3) GRPO

```bash
uv run python -m sqlrl.training.grpo_trainer
```

Saves final model to:

- `models/qwen-0.5b-reasoning-final`

## Inference

```bash
uv run python -m sqlrl.serving.inference
```

The script loads the final model and generates SQL for a sample schema/question pair.

## Training Notes and Sample Metrics

During the reinforcement learning phase, Weights & Biases logs demonstrate successful policy optimization:

- Format Adherence: `format_reward_func/mean` consistently holds at 0.95 - 1.0, proving the model internalizes the XML contract.
- Reasoning Correctness: correctness_reward_func/mean scales up to 0.8 - 1.0 over 300 steps.
- Efficiency: Completion lengths remain compact (~`55` to `70` tokens), indicating the model thinks concisely without endless looping.

This means:

- The model is learning to obey output format reliably.
- SQL correctness improves but still varies by batch.
- Optimization appears mostly stable in this snapshot.

## Troubleshooting

### `huggingface-cli` not found

Use:

```bash
uvx --from huggingface_hub hf auth login
```

### EOS token mismatch with Qwen tokenizer

Build tokenizers through `sqlrl.tokenizer.build_tokenizer`, always passing the
directory of the checkpoint you are about to run:

```python
from sqlrl.tokenizer import assert_stops_on, build_tokenizer

tokenizer = build_tokenizer("models/qwen-0.5b-sft-lora", chat=True)
assert_stops_on(tokenizer, "models/qwen-0.5b-sft-lora")
```

Setting `eos_token`/`pad_token` by hand before applying a chat template does
nothing — the template call overwrites both.

It matters more than it looks. Unsloth's `get_chat_template` rewrote the vocab
during training, so the SFT and GRPO checkpoints put `<|im_end|>` at id 151643
where stock Qwen2.5 puts `<|endoftext|>`. Load those adapters with a stock
tokenizer and generation never stops. `assert_stops_on` turns that into an
immediate error instead of a bad benchmark score.

### W&B logging errors

If `report_to="wandb"` fails, login first:

```bash
uv run wandb login
```

Or temporarily disable by changing `report_to` to `"none"` in trainer configs.
