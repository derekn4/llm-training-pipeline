# LLM Training Pipeline

An end-to-end LLM training pipeline covering all four stages of modern language model development: pretraining from scratch, supervised fine-tuning (SFT), post-training alignment (DPO), and rigorous evaluation. Built to develop hands-on experience with the full post-training stack at small scale, where every concept is identical to production pipelines but compute costs are manageable.

## Results

| Stage | Model | Dataset | Val Loss | Notes |
|-------|-------|---------|----------|-------|
| Pretrain (baseline) | 30M GPT | TinyStories | 1.6665 | 5000 steps, zero overfitting |
| Pretrain (exp1: high LR) | 30M GPT | TinyStories | 1.7782 | 2000 steps, lr=6e-4 |
| SFT | Qwen 2.5 1.5B | TBD | TBD | QLoRA, coming in Phase 2 |
| DPO | Qwen 2.5 1.5B | TBD | TBD | Coming in Phase 3 |

## Pipeline Overview

```
Phase 1: Pretraining
  TinyStories dataset → tokenize → train 30M GPT from scratch → checkpoint

Phase 2: Supervised Fine-Tuning (SFT)
  Qwen 2.5 1.5B base → QLoRA → domain instruction data → SFT checkpoint

Phase 3: Post-Training Alignment (DPO)
  SFT checkpoint → preference dataset → DPO → aligned checkpoint

Phase 4: Evaluation
  All checkpoints → lm-evaluation-harness + LLM-as-judge → results table
```

## Phase 1: Pretraining

Pretrained a GPT-style transformer from scratch on the TinyStories dataset to understand the training loop at the implementation level.

**Model config:**
- Architecture: 6 layers, 6 heads, 384 embedding dim, 256 context length
- Parameters: 30M (embedding table dominates at 50304 vocab size)
- Dataset: TinyStories (2.1M stories, ~925MB tokenized, ~328M tokens seen over 5000 steps)
- Tokenizer: GPT-2 BPE via tiktoken

**Training config:**
- Optimizer: AdamW (fused), lr=3e-4, weight decay=0.1, betas=(0.9, 0.95)
- LR schedule: linear warmup (200 steps) + cosine decay to 3e-5
- Batch: 32 sequences × 8 gradient accumulation steps × 256 tokens = 65,536 tokens/step
- Mixed precision: float16 with GradScaler
- Gradient clipping: 1.0

**Loss curve:**
```
step 0:     10.87  (random initialization, matches ln(50304) ≈ 10.82)
step 500:    2.67
step 1000:   2.28
step 2000:   1.93
step 5000:   1.67  (train: 1.6685, val: 1.6665 -- zero overfitting)
```

**Sample generation at step 5000:**

> Once upon a time there was a little girl named Anna. She was always very curious and loved to explore and see new things. One day, she decided to go for a walk in the forest. She found a big bush and decided to climb the tree. Anna started to explore the forest and soon came across a big lake...

**Experiments:**

| Run | Change | Val Loss (2000 steps) | Observation |
|-----|--------|----------------------|-------------|
| Baseline | lr=3e-4 | ~1.90 | Stable, smooth curve |
| Exp 1 | lr=6e-4 | 1.7782 | Faster early descent, slightly larger train/val gap |
| Exp 2 | grad_accum=16 (130k tokens/step) | TBD | Smoother curve, slower wall-clock |
| Exp 3 | block_size=128 | TBD | Faster per iteration, slightly higher loss |

**Key concepts demonstrated:**
- Next-token prediction as the universal pretraining objective
- Gradient accumulation to simulate large batch sizes on limited VRAM
- Cosine LR schedule with linear warmup -- why warmup prevents early instability
- Mixed precision training with GradScaler -- how float16 underflow is handled
- Weight decay applied only to 2D parameters (matrices), not biases or LayerNorm
- Flash Attention via PyTorch 2.0 scaled_dot_product_attention
- Memmap data loading -- streaming tokenized binary files without loading into RAM

## Phase 2: Supervised Fine-Tuning (SFT)

*Coming soon*

Fine-tuning Qwen 2.5 1.5B on domain-specific instruction data using QLoRA (4-bit quantization + LoRA adapters) via the `trl` SFTTrainer.

**Planned:**
- Model: Qwen/Qwen2.5-1.5B (base, not instruct)
- Method: QLoRA -- 4-bit NF4 base + LoRA rank 64 adapters
- Dataset: TBD domain instruction data
- Ablations: LoRA rank (8 vs 64), dataset size

## Phase 3: Post-Training Alignment (DPO)

*Coming soon*

Aligning the SFT checkpoint using Direct Preference Optimization -- the industry-standard alternative to RLHF PPO that eliminates the need for a separate reward model.

**Planned:**
- Method: DPO via trl DPOTrainer
- Dataset: custom preference pairs (chosen/rejected) sampled from SFT model
- Ablations: beta (KL penalty) at 0.1 vs 0.5
- Analysis: reward hacking, length bias, KL divergence from reference

## Phase 4: Evaluation

*Coming soon*

Evaluating all checkpoints with a consistent suite across standard benchmarks and a custom LLM-as-judge setup.

**Planned:**
- Standard: lm-evaluation-harness (MMLU subset, HellaSwag, ARC, HumanEval)
- Custom: 50-100 held-out domain prompts with LLM-as-judge scoring
- Judge validation: hand-grade 20 examples, report agreement rate
- Results table: base vs SFT vs DPO across all metrics

## Repository Structure

```
llm-training-pipeline/
├── data/
│   └── tinystories/
│       ├── train.bin       # 925MB tokenized training data
│       └── val.bin         # 9.5MB tokenized val data
├── checkpoints/            # saved model checkpoints (gitignored)
├── model.py                # GPT architecture (based on nanoGPT)
├── configurator.py         # config override utility (from nanoGPT)
├── 00_setup.ipynb          # environment setup and smoke test
├── 01_pretrain.ipynb       # pretraining from scratch
├── 01b_experiments.ipynb   # hyperparameter experiments
├── 02_sft.ipynb            # supervised fine-tuning
├── 03_dpo.ipynb            # DPO alignment
├── 04_eval.ipynb           # evaluation suite
└── requirements.txt
```

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/llm-training-pipeline.git

# Requirements (run in Colab or local env)
pip install -r requirements.txt
# Note: PyTorch is pre-installed on Colab with correct CUDA version
# For local: install PyTorch separately from pytorch.org
```

**Environment variables** (create a `.env` file, never commit it):
```
WANDB_API_KEY=your-wandb-key
HF_TOKEN=your-huggingface-token
```

## Compute

All training runs on Google Colab Pro (NVIDIA T4, 15GB VRAM).

| Phase | Estimated Cost |
|-------|---------------|
| Pretraining (30M, 5000 steps) | ~$2-5 |
| SFT (QLoRA, 1.5B) | ~$5-10 |
| DPO | ~$5-10 |
| Evaluation | ~$3-5 |
| **Total** | **~$15-30** |

## References

- [nanoGPT](https://github.com/karpathy/nanoGPT) -- Karpathy's minimal GPT implementation, basis for Phase 1 architecture
- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) -- dataset for pretraining
- [QLoRA paper](https://arxiv.org/abs/2305.14314) -- quantization method used in SFT
- [DPO paper](https://arxiv.org/abs/2305.18290) -- alignment method used in Phase 3
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) -- evaluation framework
