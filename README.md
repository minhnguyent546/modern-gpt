# ModernLM - Modern Language Model

<p>
  <a href="https://pytorch.org"><img alt="Torch" src="https://img.shields.io/badge/PyTorch-2.8.0+cu128-EE4C2C.svg?style=flat&logo=pytorch"></a>
  <a href="https://www.python.org"><img alt="Python" src="https://img.shields.io/badge/python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white"></a>
</p>

> Modernizing Language Model with recent advancements in architecture, optimization, and training techniques.

## Records

| Date | Key optimizations | # Tokens | Throughput (toks/s) | Val. loss | HellaSwag | OpenBookQA | ARC-e | ARC-c |
| :---: | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [v1 (2026-03-31_07-00-01)](./records/v1-2026-03-31_07-00-01/) | <ul><li>Muon optimizer ([Jordan et al., 2024](https://kellerjordan.github.io/posts/muon))</li><li>Replace learned PE with RoPE ([Su et al., 2021](https://arxiv.org/abs/2104.09864)) </li><li>Replace standard FFN with SwiGLU</li><li>Use 10x learger LR for embeddings</li><li>Use WSD ([Hu et al., 2024](https://arxiv.org/abs/2404.06395v1); [Ḧagele et al., 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/8b970e15a89bf5d12542810df8eae8fc-Paper-Conference.pdf); [Luo et al., 2025](https://arxiv.org/abs/2511.18903)) learning rate scheduling</li><li>Set bias=False</li><li>[Logits soft-capping ([Gemma Team., 2024](https://arxiv.org/pdf/2408.00118))</li><li>Flash Attention 3 ([Shah et al., 2024](https://tridao.me/publications/flash3/flash3.pdf))</li><li>RMSNorm ([Zhang et al., 2019](https://arxiv.org/abs/1910.07467))</li></ul> | 2B | 183k | 3.1388 | 30.60 | 28.80 | 47.22 | 28.75 |

## Getting started

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/minhnguyent546/modern-lm.git
   cd modern-lm
   ```

2. **Set up Python environment using uv:**
   ```bash
   # Install uv if you haven't already
   # curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install dependencies
   uv sync

   # Activate virtual environment
   source .venv/bin/activate
   ```

3. **Verify installation:**
   ```bash
   python -m modern_lm.train --help
   ```

### Dataset

> TBD

### Training

```bash
python -m modern_lm.train \
    --seed 1061109567
```
