from typing import Any, Tuple, Dict, Sequence, Optional
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
import math

IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            # Preds
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses

        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })
        # Q continue (bootstrapping target loss); Alexia: This fits Q-learning, but seems totally unecessary
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

            metrics["q_continue_loss"] = q_continue_loss.detach()
        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss), metrics, detached_outputs, new_carry.halted.all()


def _ensure_flow_matching_importable() -> None:
    try:
        import flow_matching  # noqa: F401
        return
    except ModuleNotFoundError:
        this_file = Path(__file__).resolve()
        root = this_file
        while root.name != "lms" and root.parent != root:
            root = root.parent
        fm_root = root / "repos" / "flow_matching"
        if fm_root.exists():
            sys.path.append(str(fm_root))


class DiscreteFlowMatchingLossHead(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        loss_function: str = "generalized_kl",
        source_distribution: str = "uniform",
        scheduler_type: str = "polynomial",
        scheduler_exponent: float = 2.0,
        time_epsilon: float = 1e-3,
    ):
        super().__init__()
        self.model = model
        self.time_epsilon = time_epsilon

        _ensure_flow_matching_importable()
        from flow_matching.loss import MixturePathGeneralizedKL
        from flow_matching.path import MixtureDiscreteProbPath
        from flow_matching.path.scheduler import PolynomialConvexScheduler

        if scheduler_type != "polynomial":
            raise ValueError(f"{scheduler_type} is not supported")
        self.path = MixtureDiscreteProbPath(scheduler=PolynomialConvexScheduler(n=scheduler_exponent))

        if source_distribution == "uniform":
            self.source_distribution = "uniform"
        else:
            raise ValueError(f"{source_distribution} is not supported")

        if loss_function == "generalized_kl":
            self.loss_fn = MixturePathGeneralizedKL(path=self.path, reduction="none")
            self.loss_type = "generalized_kl"
        elif loss_function == "cross_entropy":
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL_ID, reduction="none")
            self.loss_type = "cross_entropy"
        else:
            raise ValueError(f"{loss_function} is not supported")

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def _sample_x0(self, x_1: torch.Tensor) -> torch.Tensor:
        if self.source_distribution == "uniform":
            return torch.randint_like(x_1, high=self.model.config.vocab_size)  # type: ignore
        raise ValueError("Unsupported source distribution")

    def forward(
        self,
        return_keys: Sequence[str],
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        batch = model_kwargs["batch"]
        labels = batch["labels"]
        inputs = batch.get("inputs")
        puzzle_identifiers = batch["puzzle_identifiers"]

        # Replace ignore labels for path sampling
        mask = labels != IGNORE_LABEL_ID
        x_1 = torch.where(mask, labels, torch.zeros_like(labels)).to(torch.long)

        with torch.no_grad():
            x_0 = self._sample_x0(x_1)
            t = torch.rand(x_1.shape[0], device=x_1.device) * (1.0 - self.time_epsilon)
            path_sample = self.path.sample(t=t, x_0=x_0.to(torch.long), x_1=x_1)

        logits = self.model(
            x_t=path_sample.x_t,
            time=path_sample.t,
            inputs=inputs,
            puzzle_identifiers=puzzle_identifiers,
        )

        if self.loss_type == "generalized_kl":
            loss_per_token = self.loss_fn(
                logits=logits,
                x_1=x_1,
                x_t=path_sample.x_t.to(torch.long),
                t=path_sample.t,
            )
        else:
            loss_per_token = self.loss_fn(
                logits.view(-1, logits.shape[-1]),
                labels.to(torch.long).view(-1),
            ).view(labels.shape)

        loss = (loss_per_token * mask).sum() / mask.sum().clamp_min(1)

        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)
            is_correct = mask & (preds == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts

            valid_metrics = loss_counts > 0
            metrics = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "loss": loss.detach(),
            }

        detached_outputs = {"preds": preds.detach(), "logits": logits.detach()}
        detached_outputs = {k: detached_outputs[k] for k in return_keys if k in detached_outputs}

        new_carry = self.model.initial_carry(batch)  # Stateless carry
        return new_carry, loss, metrics, detached_outputs, torch.tensor(True, device=labels.device)

