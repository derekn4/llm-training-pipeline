# LLM Training Pipeline

An end-to-end LLM training pipeline covering all four stages of modern language model development: pretraining from scratch, supervised fine-tuning (SFT), post-training alignment (DPO), and rigorous evaluation. Built to develop hands-on experience with the full post-training stack at small scale, where every concept is identical to production pipelines but compute costs are manageable.

## Results

| Stage | Model | Dataset | Val Loss | Notes |
|-------|-------|---------|----------|-------|
| Pretrain (baseline) | 30M GPT | TinyStories | 1.6665 | 5000 steps, zero overfitting |
| Pretrain (exp 1: high LR) | 30M GPT | TinyStories | 1.7782 | 2000 steps, lr=6e-4 |
| SFT (LoRA r=64) | Qwen2.5-0.5B | CodeAlpaca-20k (5k subset) | 0.9122 | QLoRA, 2 epochs |
| SFT (LoRA r=8) | Qwen2.5-0.5B | CodeAlpaca-20k (5k subset) | **0.8903** | best run -- 8x fewer adapter params |
| DPO | Qwen2.5-0.5B | TBD | TBD | Coming in Phase 3 |

Val losses are only comparable *within* a stage. The pretraining and SFT numbers come from
different models, tokenizers, and objectives, so 0.89 is not "better than" 1.67.

## Pipeline Overview

```
Phase 1: Pretraining
  TinyStories dataset → tokenize → train 30M GPT from scratch → checkpoint

Phase 2: Supervised Fine-Tuning (SFT)
  Qwen2.5-0.5B base → QLoRA → CodeAlpaca instruction data → SFT checkpoint

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
| Exp 2 | grad_accum=16 (131k tokens/step) | 1.8524 | Beats baseline, but saw 2x the tokens -- confounded, see note |
| Exp 3 | block_size=128 | 2.0949 | Faster per iteration, clearly worse loss -- less context to predict from |

All runs are capped at 2000 optimizer steps, which makes them steps-matched but **not**
compute-matched -- and that confounds Exp 2. Doubling gradient accumulation doubles
tokens per step, so Exp 2 saw ~262M tokens against the baseline's ~131M. Its better loss
is partly just more data, not purely a larger-batch effect. Exp 3 is the cleaner result:
halving the context to 128 tokens is strictly less to condition on, and the loss degrades
accordingly.

**Key concepts demonstrated:**
- Next-token prediction as the universal pretraining objective
- Gradient accumulation to simulate large batch sizes on limited VRAM
- Cosine LR schedule with linear warmup -- why warmup prevents early instability
- Mixed precision training with GradScaler -- how float16 underflow is handled
- Weight decay applied only to 2D parameters (matrices), not biases or LayerNorm
- Flash Attention via PyTorch 2.0 scaled_dot_product_attention
- Memmap data loading -- streaming tokenized binary files without loading into RAM

## Phase 2: Supervised Fine-Tuning (SFT)

Fine-tuned Qwen2.5-0.5B (base, not instruct, so the SFT effect is visible) on code
instruction data using QLoRA -- a 4-bit NF4 base with trainable LoRA adapters -- via
`trl`'s `SFTTrainer`.

**Setup:**
- Model: `Qwen/Qwen2.5-0.5B`, 4-bit NF4 with double quantization, fp16 compute
- Adapters: LoRA on all attention and MLP projections (`q/k/v/o_proj`, `gate/up/down_proj`)
- Dataset: CodeAlpaca-20k, 5000 train / 1002 val (5% held out, seed 42)
- Prompt format: Alpaca-style `### Instruction:` / `### Response:`
- Training: 2 epochs, lr 2e-4 cosine with warmup, batch 2 x grad accum 16 (32 effective), `max_length=256`, `adamw_torch`
- Hardware: single T4, 1.5-1.7 GB peak VRAM, ~56 min per full run

**Baseline run (LoRA r=64):**
```
step   train    eval
  50  0.9639       -
 100  0.8790       -
 150  0.8674       -
 200  0.7031  0.9200
 250  0.6819       -
 300  0.6728       -
 314       -  0.9122   <- best
```

**Ablations:**

| Run | LoRA r | Train examples | Trainable params | Best eval loss | Final train loss | Train/eval gap |
|-----|--------|----------------|------------------|----------------|------------------|----------------|
| Baseline | 64 | 5000 | 35,192,832 (6.65%) | 0.9122 | 0.6728 | +0.2394 |
| Rank ablation | 8 | 5000 | 4,399,104 (0.83%) | **0.8903** | 0.7843 | **+0.1060** |
| Data ablation | 64 | 500 | 35,192,832 (6.65%) | 0.9495 | 0.6299 | +0.3196 |

**Rank 8 beat rank 64 with 8x fewer adapter parameters.** The train/eval gap explains why:
r=64 puts 6.65% of a 0.5B model into trainable adapters, which is more capacity than 5000
examples can support, so the surplus goes into memorizing the training set -- train loss
0.67 against eval 0.91. Rank 8 fits the training data *worse* (0.78) and generalizes
*better* (0.89). This is the low-intrinsic-dimensionality argument from the LoRA paper,
plus a regularization effect from simply having less capacity to overfit with.

**The 500-example run is a textbook overfitting curve.** Lowest train loss, worst eval
loss, widest gap -- and its best eval landed at step 15 of 32, meaning val loss turned
upward halfway through and kept climbing.

*Caveat on the rank result:* the baseline was measured at only 2 eval points (`eval_steps=200`
on a 314-step run) while the ablations use ~8, and every run is a single seed. A 0.022 delta
between an under-sampled curve and a well-sampled one is suggestive, not conclusive.

**Base vs SFT:** 20 fixed held-out prompts -- 15 code generation, 3 explanation, 1
deliberately underspecified, 1 ill-posed -- run through both models with greedy decoding
and an identical template. Full side-by-side output in `results/sft_qualitative.md`.

| Metric | Base | SFT |
|--------|------|-----|
| Mean response length (chars) | 304 | 160 |
| Ran on into a hallucinated next `### Instruction:` block | 0/20 | 0/20 |

The leak metric found nothing, which is itself worth reporting: Qwen2.5-0.5B base already
terminates on its own in this format rather than rambling, so "SFT taught the model to
stop" is *not* what happened here. The measurable change is response length -- SFT answers
are about half as long, consistent with terse on-task completions replacing discursive ones.

Sample SFT output:
```
### Instruction:
Write a Python function that takes a list of numbers and returns the sum of all even numbers.

### Response:
def even_sum(numbers):
    even_sum = 0
    for num in numbers:
        if num % 2 == 0:
            even_sum += num
    return even_sum
```

**Design decisions:**
- **Alpaca template, not Qwen's chat template.** The base checkpoint has no instruction-tuned
  chat semantics to inherit, so a plain delimiter format keeps the comparison honest and
  avoids a train/eval formatting mismatch -- the failure mode that silently destroys SFT
  performance when the fine-tune and the eval disagree on special tokens.
- **Loss over the full sequence, not completion-only.** TRL's default with `dataset_text_field`
  trains on instruction tokens too. Acceptable here since instructions are short relative to
  responses, but masking the prompt is the more standard choice.
- **Eval frequency scaled to run length** in the ablation harness. The baseline's
  `eval_steps=200` produced two points on a 314-step run, too coarse to see where val loss
  turns -- which is exactly what the 500-example run needed in order to be interpretable.

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
├── results/                # metrics and generations (committed)
│   ├── sft_r64_log.json    # baseline train/eval loss history
│   ├── sft_*_log.json      # one per ablation run
│   ├── sft_ablations.md    # ablation comparison table
│   └── sft_qualitative.md  # 20-prompt base vs SFT side-by-side
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
| SFT (QLoRA, 0.5B, 3 runs) | ~$5-10 |
| DPO | ~$5-10 |
| Evaluation | ~$3-5 |
| **Total** | **~$15-30** |

## References

- [nanoGPT](https://github.com/karpathy/nanoGPT) -- Karpathy's minimal GPT implementation, basis for Phase 1 architecture
- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) -- dataset for pretraining
- [QLoRA paper](https://arxiv.org/abs/2305.14314) -- quantization method used in SFT
- [DPO paper](https://arxiv.org/abs/2305.18290) -- alignment method used in Phase 3
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) -- evaluation framework
