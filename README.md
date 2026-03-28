# ModernLM - Modern Language Model

<p>
  <a href="https://pytorch.org"><img alt="Torch" src="https://img.shields.io/badge/PyTorch-2.8.0+cu128-EE4C2C.svg?style=flat&logo=pytorch"></a>
  <a href="https://www.python.org"><img alt="Python" src="https://img.shields.io/badge/python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white"></a>
</p>

> Modernizing Language Model with recent advancements in architecture, optimization, and training techniques.

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
