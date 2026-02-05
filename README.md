# Exploring Reasoning with Flow Matching

**Research Internship Project** - Testing the hypothesis that Tiny Reasoning Models (TRM) are equivalent to Flow Matching models for solving hard reasoning tasks.

## 🎯 Research Question

Can we replace the **Tiny Reasoning Model (TRM)** architecture with a **Discrete Flow Matching** model and achieve similar performance on hard reasoning tasks like Sudoku and Maze solving?

### Hypothesis

The reasoning model in [TRM (arXiv:2510.04871)](https://arxiv.org/abs/2510.04871) that learns to solve hard Sudoku or mazes with ~1000 samples may be equivalent to a flow matching model.

## 📁 Project Structure

```
.
├── repos/
│   ├── trm/
│   │   └── TinyRecursiveModels/     # Original TRM codebase
│   │       ├── models/
│   │       │   ├── flow_matching.py  # ✨ NEW: Discrete Flow Matching model
│   │       │   └── losses.py          # ✨ UPDATED: Flow matching loss head
│   │       ├── config/
│   │       │   └── arch/
│   │       │       └── flow_matching.yaml  # ✨ NEW: Flow matching config
│   │       └── pretrain.py            # Training script (supports both TRM & FM)
│   └── flow_matching/                 # Flow matching reference implementation
├── experiment/
│   ├── inspect_data.py               # Dataset inspection utilities
│   └── trm_sanity/
│       └── trm_forward_sanity.py     # TRM model import sanity checks
└── flowmatching.ipynb                # Flow matching exploration notebook
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n fm_reasoning python=3.10
conda activate fm_reasoning

# Install PyTorch (adjust CUDA version as needed)
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126

# Install dependencies
cd repos/trm/TinyRecursiveModels
pip install -r requirements.txt

# Install flow matching package (if needed)
# The code automatically finds repos/flow_matching/ if available
```

### 2. Dataset Preparation

```bash
cd repos/trm/TinyRecursiveModels

# Generate Sudoku dataset
python -m dataset.build_sudoku_dataset \
  --output-dir data/sudoku-extreme-1k \
  --num-puzzles 1000

# Generate Maze dataset  
python -m dataset.build_maze_dataset \
  --output-dir data/maze-30x30-hard-1k \
  --num-puzzles 1000
```

### 3. Training

#### Train with TRM (baseline)
```bash
cd repos/trm/TinyRecursiveModels
WANDB_MODE=disabled python pretrain.py \
  arch=trm \
  data_paths=['data/sudoku-extreme-1k'] \
  global_batch_size=32 \
  epochs=100
```

#### Train with Flow Matching
```bash
cd repos/trm/TinyRecursiveModels
WANDB_MODE=disabled python pretrain.py \
  arch=flow_matching \
  data_paths=['data/sudoku-extreme-1k'] \
  global_batch_size=32 \
  epochs=100
```

## 🔬 What Was Added

### 1. Discrete Flow Matching Model (`models/flow_matching.py`)

A time-conditioned transformer that:
- Takes noisy tokens `x_t` and conditioning inputs (puzzle state)
- Uses sinusoidal time embeddings
- Predicts logits for the target distribution `p(x_1 | x_t, t)`
- Compatible with the TRM training pipeline

### 2. Flow Matching Loss Head (`models/losses.py`)

`DiscreteFlowMatchingLossHead` that:
- Samples `x_t` from a discrete flow path (mixture of uniform and target)
- Computes generalized KL divergence loss
- Supports both `generalized_kl` and `cross_entropy` loss functions
- Integrates seamlessly with existing training loop

### 3. Architecture Config (`config/arch/flow_matching.yaml`)

Configuration for flow matching model:
- 6-layer transformer with 512 hidden size
- 8 attention heads
- RoPE positional encodings
- Polynomial scheduler (exponent=2.0) for flow path

## 📊 Current Status

✅ **Completed:**
- Flow matching model implementation
- Integration with TRM training pipeline
- Smoke tests on Sudoku dataset (both TRM and FM train successfully)
- Initial performance comparison (5 epochs)

🔄 **In Progress:**
- Longer training runs for fair comparison
- Evaluation metrics (exact solve rate, token accuracy)
- Hyperparameter tuning for flow matching

📋 **Next Steps:**
- Add proper evaluators for Sudoku/Maze
- Compare learning curves (TRM vs Flow Matching)
- Test on Maze dataset
- Analyze if flow matching can match TRM's sample efficiency

## 🧪 Experiments

### Sanity Checks

```bash
# Verify TRM model can be imported
python experiment/trm_sanity/trm_forward_sanity.py

# Inspect dataset format
python experiment/inspect_data.py
```

### Training Comparison

Run both models with identical hyperparameters and compare:
- Training loss curves
- Exact solve rate (all tokens correct)
- Token-level accuracy
- Sample efficiency (convergence speed)

## 📚 References

- **TRM Paper:** [Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/abs/2510.04871)
- **Flow Matching:** [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- **Discrete Flow Matching:** [Discrete Flow Matching](https://arxiv.org/abs/2412.06264)
- **TRM Codebase:** [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)
- **Flow Matching Library:** [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching)

## 👤 Author

**Prodipto** - Research Internship  
Supervisor: Michele De Vita

## 📝 License

This project builds upon the TRM codebase. Please refer to the original TRM repository for licensing information.

---

**Note:** This is a research project exploring the equivalence between recursive reasoning models and flow matching. Results and conclusions are preliminary and subject to further validation.
