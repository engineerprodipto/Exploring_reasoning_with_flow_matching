from typing import Dict, Optional
import os

import torch
import torch.distributed as dist

from dataset.common import PuzzleDatasetMetadata

IGNORE_LABEL_ID = -100


class SudokuMazeEvaluator:
    """
    Evaluator for Sudoku and Maze datasets.
    
    Computes:
    - Exact solve rate: fraction of puzzles where all tokens are correct
    - Token accuracy: fraction of non-ignored tokens that are correct
    """
    
    required_outputs = {"preds"}
    
    def __init__(
        self,
        data_path: str,
        eval_metadata: PuzzleDatasetMetadata,
        **kwargs
    ):
        self.data_path = data_path
        self.eval_metadata = eval_metadata
        self.ignore_label_id = eval_metadata.ignore_label_id or IGNORE_LABEL_ID
        
        # Accumulated metrics
        self.total_exact_solves = 0
        self.total_puzzles = 0
        self.total_correct_tokens = 0
        self.total_valid_tokens = 0
        
    def begin_eval(self):
        """Reset evaluation state."""
        self.total_exact_solves = 0
        self.total_puzzles = 0
        self.total_correct_tokens = 0
        self.total_valid_tokens = 0
    
    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        """
        Process a batch of predictions.
        
        Args:
            batch: Contains 'labels' and other batch data
            preds: Contains 'preds' (model predictions)
        """
        labels = batch["labels"].cpu()  # [B, seq_len]
        preds_tokens = preds["preds"].cpu()  # [B, seq_len]
        
        # Create mask for valid tokens (non-ignored)
        valid_mask = labels != self.ignore_label_id
        
        # Compute exact solves: all valid tokens match
        # For each sequence, check if all valid tokens are correct
        correct_tokens = (preds_tokens == labels) & valid_mask
        all_valid_correct = (correct_tokens.sum(dim=-1) == valid_mask.sum(dim=-1))
        
        self.total_exact_solves += all_valid_correct.sum().item()
        self.total_puzzles += labels.shape[0]
        
        # Compute token-level accuracy
        self.total_correct_tokens += correct_tokens.sum().item()
        self.total_valid_tokens += valid_mask.sum().item()
    
    def result(
        self,
        save_path: Optional[str],
        rank: int,
        world_size: int,
        group: Optional[torch.distributed.ProcessGroup] = None
    ) -> Optional[Dict[str, float]]:
        """
        Compute final evaluation metrics.
        
        Returns:
            Dictionary with 'exact_solve_rate' and 'token_accuracy'
        """
        # Gather metrics from all ranks
        if world_size > 1:
            metrics_tensor = torch.tensor([
                self.total_exact_solves,
                self.total_puzzles,
                self.total_correct_tokens,
                self.total_valid_tokens
            ], dtype=torch.float64, device="cuda")
            
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM, group=group)
            
            total_exact_solves = metrics_tensor[0].item()
            total_puzzles = metrics_tensor[1].item()
            total_correct_tokens = metrics_tensor[2].item()
            total_valid_tokens = metrics_tensor[3].item()
        else:
            total_exact_solves = self.total_exact_solves
            total_puzzles = self.total_puzzles
            total_correct_tokens = self.total_correct_tokens
            total_valid_tokens = self.total_valid_tokens
        
        # Only rank 0 computes and returns results
        if rank != 0:
            return None
        
        # Compute metrics
        exact_solve_rate = total_exact_solves / max(total_puzzles, 1)
        token_accuracy = total_correct_tokens / max(total_valid_tokens, 1)
        
        results = {
            "exact_solve_rate": exact_solve_rate,
            "token_accuracy": token_accuracy,
            "total_puzzles": total_puzzles,
            "total_exact_solves": total_exact_solves,
        }
        
        # Save results if requested
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            with open(save_path, "w") as f:
                import json
                json.dump(results, f, indent=2)
        
        return results
