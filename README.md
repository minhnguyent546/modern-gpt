# ModernLM - Modern Language Model

<p>
  <a href="https://pytorch.org"><img alt="Torch" src="https://img.shields.io/badge/PyTorch-2.8.0+cu128-EE4C2C.svg?style=flat&logo=pytorch"></a>
  <a href="https://www.python.org"><img alt="Python" src="https://img.shields.io/badge/python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white"></a>
</p>

> Modernizing Language Model with recent advancements in architecture, optimization, and training techniques.

## Records

| Date | Key optimizations | # Tokens | Throughput (toks/s) | Val. loss | HellaSwag | OpenBookQA | ARC-e | ARC-c |
| :---: | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [v3 (2026-04-05_14-38-42)](./records/v3-2026-04-05_14-38-42/) | [Details](./records/v3-2026-04-05_14-38-42/) | 2B | 138k | 2.8207 | 33.58 | 27.60 | 45.08 | 26.88 |
| [v2 (2026-04-04_08-38-15)](./records/v2-2026-04-04_08-38-15/) | [Details](./records/v2-2026-04-04_08-38-15/) | 2B | 138k | 3.0862 | 31.10 | 29.60 | 47.73 | 28.50 |
| [v1 (2026-03-31_07-00-01)](./records/v1-2026-03-31_07-00-01/) | [Details](./records/v1-2026-03-31_07-00-01/) | 2B | 183k | 3.1388 | 30.60 | 28.80 | 47.22 | 28.75 |

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

> See [./records](./records).

### Training

> Detail training scripts can be found in [./records](./records).

```bash
python -m modern_lm.train \
    --seed 1061109567
```
