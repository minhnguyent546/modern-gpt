# Modern Language Modeling

> Modernizing language modeling with recent advancements.

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

The dataset used for training can be found [here on HF](minhnguyent546/fineweb-edu-10BT-for-gpt2).

### Training

```bash
python -m modern_lm.train \
    --seed 42
```
