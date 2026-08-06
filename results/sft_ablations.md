# SFT ablations

Base model `Qwen/Qwen2.5-0.5B` in 4-bit NF4, CodeAlpaca-20k, 2 epochs, lr 2e-4 cosine with proportional warmup, max_length 256, batch 2 x grad_accum 16, seed 42.
Only LoRA rank and training-set size vary; the 1002-example val split is identical across runs.

| Run | LoRA r | Train examples | Trainable params | Best eval loss | Final train loss | Peak VRAM (GB) | Wall clock (min) |
|---|---|---|---|---|---|---|---|
| sft-r64-n5000 (baseline) | 64 | 5000 | 35,192,832 | 0.9122 | 0.6728 | - | - |
| sft-r8-n5000 | 8 | 5000 | 4,399,104 | 0.8903 | 0.7843 | 1.46 | 56.3 |
| sft-r64-n500 | 64 | 500 | 35,192,832 | 0.9495 | 0.6299 | 1.70 | 13.1 |

Train/eval gap at end of training:

- `sft-r64-n5000 (baseline)`: +0.2394
- `sft-r8-n5000`: +0.1060
- `sft-r64-n500`: +0.3196
