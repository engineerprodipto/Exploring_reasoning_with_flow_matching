# Evaluation Guide

## Sudoku/Maze Evaluator

The `SudokuMazeEvaluator` computes two key metrics for Sudoku and Maze datasets:

1. **Exact Solve Rate**: Fraction of puzzles where all tokens are correct
2. **Token Accuracy**: Fraction of non-ignored tokens that are correct

## Usage

### Option 1: Using config file

```bash
cd repos/trm/TinyRecursiveModels
WANDB_MODE=disabled python pretrain.py \
  --config-path config \
  --config-name cfg_sudoku_maze \
  arch=trm  # or arch=flow_matching
```

### Option 2: Command-line override

```bash
cd repos/trm/TinyRecursiveModels
WANDB_MODE=disabled python pretrain.py \
  arch=trm \
  data_paths=['data/sudoku-extreme-1k'] \
  data_paths_test=['data/sudoku-extreme-1k'] \
  evaluators='[{name: sudoku_maze@SudokuMazeEvaluator}]' \
  global_batch_size=32 \
  epochs=100 \
  eval_interval=10 \
  min_eval_interval=0
```

## Output Metrics

The evaluator returns:
- `exact_solve_rate`: Percentage of puzzles solved completely (all tokens correct)
- `token_accuracy`: Percentage of individual tokens predicted correctly
- `total_puzzles`: Total number of puzzles evaluated
- `total_exact_solves`: Number of puzzles solved exactly

These metrics are logged to WandB (if enabled) and printed during evaluation.

## Example Output

```
Processing batch 1: train
  Completed inference in 1 steps
...
Evaluation Results:
  exact_solve_rate: 0.15 (15%)
  token_accuracy: 0.85 (85%)
  total_puzzles: 1000
  total_exact_solves: 150
```

## Notes

- The evaluator works with both TRM and Flow Matching models
- It automatically handles distributed evaluation (multi-GPU)
- Results are aggregated across all ranks before computing final metrics
- Only rank 0 prints/returns the final results
