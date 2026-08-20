"""The LTE-IMG-NoT causal state update.

The module intentionally has no timestep input.  Its state at ``t`` depends
only on the ResNet feature at ``t`` and the preceding latent state.
"""
from typing import Optional

import torch
import torch.nn as nn


class ImageNoTimeTemporalEncoder(nn.Module):
    """Small recurrent residual MLP for causal ResNet-feature histories."""

    def __init__(self, image_embedding_dim: int, latent_dim: int = 32,
                 hidden_dim: int = 128, recurrent: bool = True):
        super().__init__()
        self.image_embedding_dim = int(image_embedding_dim)
        self.latent_dim = int(latent_dim)
        self.recurrent = bool(recurrent)
        input_dim = self.image_embedding_dim + (self.latent_dim if recurrent else 0)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.latent_dim),
        )

    def initial_latent(self, batch_size: int, *, device=None, dtype=None):
        return torch.zeros(batch_size, self.latent_dim, device=device, dtype=dtype)

    def forward(self, image_embedding_t: torch.Tensor,
                z_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        if image_embedding_t.ndim != 2 or image_embedding_t.shape[-1] != self.image_embedding_dim:
            raise ValueError(
                "image_embedding_t must have shape (B, image_embedding_dim)"
            )
        if not self.recurrent:
            return torch.tanh(self.mlp(image_embedding_t))
        if z_prev is None:
            z_prev = self.initial_latent(
                image_embedding_t.shape[0], device=image_embedding_t.device,
                dtype=image_embedding_t.dtype,
            )
        if z_prev.shape != (image_embedding_t.shape[0], self.latent_dim):
            raise ValueError("z_prev has the wrong shape")
        return torch.tanh(z_prev + self.mlp(torch.cat([image_embedding_t, z_prev], dim=-1)))

    def encode_history(self, image_embeddings: torch.Tensor,
                       valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode padded episode prefixes while preserving state after padding."""
        if image_embeddings.ndim != 3 or image_embeddings.shape[-1] != self.image_embedding_dim:
            raise ValueError("image_embeddings must have shape (B, T, image_embedding_dim)")
        batch_size, history_len = image_embeddings.shape[:2]
        if valid_mask is None:
            valid_mask = torch.ones(
                batch_size, history_len, dtype=torch.bool, device=image_embeddings.device
            )
        z_prev = self.initial_latent(
            batch_size, device=image_embeddings.device, dtype=image_embeddings.dtype
        )
        states = []
        for index in range(history_len):
            z_candidate = self(image_embeddings[:, index], z_prev)
            z_prev = torch.where(valid_mask[:, index, None], z_candidate, z_prev)
            states.append(z_prev)
        return torch.stack(states, dim=1)


def gather_history_latents(latents: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather per-observation LTE states from padded prefix states."""
    return torch.gather(
        latents, 1,
        indices.long().unsqueeze(-1).expand(-1, -1, latents.shape[-1]),
    )
