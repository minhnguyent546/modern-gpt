# v2 (2026-04-04_08-38-15)

## Key optimizations

- Use deeper model with 30 layers, 576 hidden size, 9 attention heads (3 kv heads), and 1536 intermediate size, resulting in 134.48M parameters.
- Long-Short Sliding Window Attention with global:local ratio set to 4:1 inspired by [Gemma 2](https://arxiv.org/abs/2408.00118)
- Better weights initialization:
  + LayerNorm Scaling ([Sun et al., 2025](https://arxiv.org/abs/2502.05795))
  + Use Truncated normal $(\pm 3\sigma)$ for stability, following OLMo 2 ([Walsh et al., 2025](https://arxiv.org/abs/2501.00656)).
  + Width-aware std for linear layers ([Yang et al., 2022](https://arxiv.org/abs/2203.03466); [Lingle et al., 2024](https://arxiv.org/abs/2404.05728v5); [Groeneveld et al., 2024](https://arxiv.org/abs/2402.00838))
- Grouped-Query Attention ([Ainslie et al., 2023](https://arxiv.org/abs/2305.13245))

## Model and training information

- Hardware: single node with 2x4090 GPUs
- Model size: 134.48M parameters
- Dataset: Fineweb-Edu 10BT subset
- Batch size: 64
- Sequence length: 8192 (window size 512)
- Training time 4h 24m 24.88s
- Number of tokens seen: 2B
- Peak VRAM usage: 14062.38 MB

## Results

| Metric | Result | Previous result ([v1](../v1-2026-03-31_07-00-01/)) |
| --- | ---: | ---: |
| Throughput | 138k tokens/s |**183k tokens/s** |
| Validation loss | **3.0862** | 3.1388 |
| HellaSwag | **31.10** | 30.60 |
| OpenBookQA | **29.60** | 28.80 |
| ARC-easy | **47.73** | 47.22 |
| ARC-challenge | 28.50 | **28.75** |
