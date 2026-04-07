# v3 (2026-04-05_14-38-42)

## Key optimizations

- Switch from FineWeb-Edu to **Nemotron-ClimbMix** dataset ([Diao et al., 2025](https://arxiv.org/abs/2504.13161))

## Model and training information

- Hardware: single node with 2x4090 GPUs
- Model size: 134.48M parameters
- Dataset: Nemotron-ClimbMix
- Batch size: 64
- Sequence length: 8192 (window size 512)
- Training time 4h 27m 43.89s
- Number of tokens seen: 2B
- Peak VRAM usage: 14052.28 MB

## Results

| Metric | Result | Previous result ([v2](../v2-2026-04-04_08-38-15/)) |
| --- | ---: | ---: |
| Throughput | 138k tokens/s | 138k tokens/s |
| Validation loss | **2.8207** | 3.0862 |
| HellaSwag | **33.58** | 31.10 |
| OpenBookQA | 27.60 | **29.60** |
| ARC-easy | 45.08 | **47.73** |
| ARC-challenge | 26.88 | 28.50 |
