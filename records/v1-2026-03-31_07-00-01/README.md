# v1 (2026-03-31_07-00-01)

## Model and training information

- Model size: 122.68M parameters
- Dataset: Fineweb-Edu 10BT subset
- Number of training tokens: 2B
- Batch size: 256
- Sequence length: 2048

## Results

- Throughput: 183k tokens/s (2x4090)
- Validation loss: 3.1388
- HellaSwag: 30.60
- OpenBookQA: 28.80
- ARC-easy: 47.22
- ARC-challenge: 28.75

## Key optimizations

- Muon optimizer ([Jordan et al., 2024](https://kellerjordan.github.io/posts/muon))
- Replace learned PE with RoPE ([Su et al., 2021](https://arxiv.org/abs/2104.09864))
- Replace standard FFN with SwiGLU
- Use 10x learger LR for embeddings
- Use WSD ([Hu et al., 2024](https://arxiv.org/abs/2404.06395v1); [Ḧagele et al., 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/8b970e15a89bf5d12542810df8eae8fc-Paper-Conference.pdf); [Luo et al., 2025](https://arxiv.org/abs/2511.18903)) learning rate scheduling
- Set bias=False
- Logits soft-capping ([Gemma Team., 2024](https://arxiv.org/pdf/2408.00118))
- Flash Attention 3 ([Shah et al., 2024](https://tridao.me/publications/flash3/flash3.pdf))
- RMSNorm ([Zhang et al., 2019](https://arxiv.org/abs/1910.07467))
