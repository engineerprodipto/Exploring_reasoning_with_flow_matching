from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import (
    rms_norm,
    Attention,
    RotaryEmbedding,
    CastedEmbedding,
    CastedLinear,
    SwiGLU,
)
from models.sparse_embedding import CastedSparseEmbedding


@dataclass
class FlowMatchingCarry:
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class FlowMatchingConfig(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    # Transformer config
    hidden_size: int
    num_layers: int
    num_heads: int
    expansion: float
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Time embedding
    time_emb_dim: int = 256

    forward_dtype: str = "bfloat16"


def _timestep_embedding(time: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    # time: [B] in [0, 1]
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=time.device) / half
    )
    args = time[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class FlowMatchingBlock(nn.Module):
    def __init__(self, config: FlowMatchingConfig) -> None:
        super().__init__()
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False,
        )
        self.mlp = SwiGLU(hidden_size=config.hidden_size, expansion=config.expansion)
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: Tuple[torch.Tensor, torch.Tensor], hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps,
        )
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class DiscreteFlowMatchingModel(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = FlowMatchingConfig(**config_dict)
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)

        self.time_embed = nn.Sequential(
            nn.Linear(self.config.time_emb_dim, self.config.hidden_size),
            nn.SiLU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
        )

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_ndim > 0 else 0
        if self.config.puzzle_emb_ndim > 0:
            self._puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )
        else:
            self._puzzle_emb = None

        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta,
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype,
            )
        else:
            raise NotImplementedError()

        self.layers = nn.ModuleList([FlowMatchingBlock(self.config) for _ in range(self.config.num_layers)])

    @property
    def puzzle_emb(self):
        return self._puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> FlowMatchingCarry:
        batch_size = batch["inputs"].shape[0]
        return FlowMatchingCarry(
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=batch["inputs"].device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=batch["inputs"].device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def _input_embeddings(self, x_t: torch.Tensor, inputs: Optional[torch.Tensor], puzzle_identifiers: torch.Tensor) -> torch.Tensor:
        # Token embedding (x_t + conditioning inputs)
        embedding = self.embed_tokens(x_t.to(torch.int32))
        if inputs is not None:
            embedding = embedding + self.embed_tokens(inputs.to(torch.int32))

        # Puzzle embeddings (prefix tokens)
        if self.config.puzzle_emb_ndim > 0 and self._puzzle_emb is not None:
            puzzle_embedding = self._puzzle_emb(puzzle_identifiers)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding),
                dim=-2,
            )

        # Position embeddings
        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        return self.embed_scale * embedding

    def forward(self, x_t: torch.Tensor, time: torch.Tensor, inputs: Optional[torch.Tensor], puzzle_identifiers: torch.Tensor) -> torch.Tensor:
        # Build embeddings
        input_embeddings = self._input_embeddings(x_t, inputs, puzzle_identifiers)
        time_emb = self.time_embed(_timestep_embedding(time, self.config.time_emb_dim)).to(self.forward_dtype)
        hidden_states = input_embeddings + time_emb[:, None, :]

        cos_sin = self.rotary_emb() if hasattr(self, "rotary_emb") else None
        for layer in self.layers:
            hidden_states = layer(cos_sin=cos_sin, hidden_states=hidden_states)

        logits = self.lm_head(hidden_states)[:, self.puzzle_emb_len :]
        return logits
