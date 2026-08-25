"""The LTE-IMG-NoT causal state update and training-only history objective.

The module intentionally has no timestep input.  Its state at ``t`` depends
only on the ResNet feature at ``t`` and the preceding latent state.
"""
from typing import Optional, Sequence

import torch
import torch.nn as nn


class ImageNoTimeTemporalEncoder(nn.Module):
    """Small recurrent residual MLP for causal ResNet-feature histories."""

    def __init__(self, image_embedding_dim: int, latent_dim: int = 32,
                 hidden_dim: int = 128, recurrent: bool = True,
                 num_hidden_layers: int = 1):
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be at least one")
        self.image_embedding_dim = int(image_embedding_dim)
        self.latent_dim = int(latent_dim)
        self.recurrent = bool(recurrent)
        self.num_hidden_layers = int(num_hidden_layers)
        input_dim = self.image_embedding_dim + (self.latent_dim if recurrent else 0)
        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(self.num_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, self.latent_dim))
        self.mlp = nn.Sequential(*layers)

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


class ImageHistoryDecoder(nn.Module):
    """Reconstruct a past ResNet embedding from an LTE state and a lag.

    This is the image-input counterpart of the original training-only
    ``HistoryDecoder``.  It is deliberately not used during rollout.
    """

    def __init__(self, latent_dim: int, image_embedding_dim: int,
                 hidden_dim: int = 128, num_hidden_layers: int = 1):
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be at least one")
        self.num_hidden_layers = int(num_hidden_layers)
        layers = [nn.Linear(latent_dim + 1, hidden_dim), nn.SiLU()]
        for _ in range(self.num_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, image_embedding_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, latents: torch.Tensor, query_lags: torch.Tensor) -> torch.Tensor:
        if query_lags.shape[-1] != 1:
            raise ValueError("query_lags must have final dimension 1")
        return self.mlp(torch.cat([latents, query_lags], dim=-1))


class MultiImageEmbeddingFusion(nn.Module):
    """Fuse any number of per-camera features into one LTE input feature."""

    def __init__(
        self, embedding_dims: Sequence[int], output_dim: int, hidden_dim: int
    ):
        super().__init__()
        if len(embedding_dims) < 2:
            raise ValueError("MultiImageEmbeddingFusion requires at least two views")
        self.embedding_dims = tuple(int(dim) for dim in embedding_dims)
        if any(dim <= 0 for dim in self.embedding_dims):
            raise ValueError("image embedding dimensions must be positive")
        self.output_dim = int(output_dim)
        if self.output_dim <= 0 or hidden_dim <= 0:
            raise ValueError("fusion dimensions must be positive")
        self.mlp = nn.Sequential(
            nn.Linear(sum(self.embedding_dims), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), self.output_dim),
        )

    def forward(self, embeddings: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(embeddings) != len(self.embedding_dims):
            raise ValueError("number of image embeddings does not match configured views")
        prefix = embeddings[0].shape[:-1]
        for embedding, expected_dim in zip(embeddings, self.embedding_dims):
            if embedding.shape[:-1] != prefix or embedding.shape[-1] != expected_dim:
                raise ValueError("image embeddings have incompatible shapes")
        return self.mlp(torch.cat(list(embeddings), dim=-1))


class MultiImageHistoryDecoder(nn.Module):
    """Training-only shared history decoder trunk with per-camera heads."""

    def __init__(
        self, latent_dim: int, image_embedding_dims: Sequence[int], hidden_dim: int,
        num_hidden_layers: int = 1,
    ):
        super().__init__()
        if len(image_embedding_dims) < 2:
            raise ValueError("MultiImageHistoryDecoder requires at least two views")
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be at least one")
        self.image_embedding_dims = tuple(int(dim) for dim in image_embedding_dims)
        layers = [nn.Linear(latent_dim + 1, hidden_dim), nn.SiLU()]
        for _ in range(num_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleList(
            nn.Linear(hidden_dim, dim) for dim in self.image_embedding_dims
        )

    def forward(self, latents: torch.Tensor, query_lags: torch.Tensor) -> list[torch.Tensor]:
        if query_lags.shape[-1] != 1:
            raise ValueError("query_lags must have final dimension 1")
        hidden = self.trunk(torch.cat([latents, query_lags], dim=-1))
        return [head(hidden) for head in self.heads]


def _sample_stratified_history_lags(elapsed: torch.Tensor, num_queries: int) -> torch.Tensor:
    """Sample one lag from every equal-width interval in the causal past."""
    if num_queries <= 0:
        raise ValueError("num_history_queries must be positive")
    bin_offsets = torch.arange(
        num_queries, device=elapsed.device, dtype=elapsed.dtype
    ) / num_queries
    random_offsets = torch.rand(
        *elapsed.shape, num_queries, device=elapsed.device, dtype=elapsed.dtype
    ) / num_queries
    return (bin_offsets + random_offsets) * elapsed.unsqueeze(-1)


def _interpolate_past_embeddings(
    history_embeddings: torch.Tensor,
    current_indices: torch.Tensor,
    query_lags: torch.Tensor,
) -> torch.Tensor:
    """Linearly interpolate embeddings at causal, episode-local frame lags."""
    batch_size, history_len, embedding_dim = history_embeddings.shape
    query_positions = current_indices.to(query_lags.dtype).unsqueeze(-1) - query_lags
    # The original NoT path derives its reconstruction coordinate from
    # ``arange(episode_length)``.  Hence position and index are identical.
    upper = query_positions.ceil().long()
    upper_bound = current_indices.long().unsqueeze(-1).expand_as(query_positions)
    upper = torch.minimum(upper, upper_bound).clamp(min=0, max=history_len - 1)
    lower = (upper - 1).clamp(min=0)

    def gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return torch.gather(
            values,
            1,
            indices.reshape(batch_size, -1).unsqueeze(-1).expand(-1, -1, embedding_dim),
        ).reshape(*indices.shape, embedding_dim)

    low = gather(history_embeddings, lower)
    high = gather(history_embeddings, upper)
    weight = (query_positions - lower.to(query_positions.dtype)).unsqueeze(-1)
    return low + weight * (high - low)


def image_history_reconstruction_loss(
    decoder: ImageHistoryDecoder,
    observation_latents: torch.Tensor,
    history_embeddings: torch.Tensor,
    observation_indices: torch.Tensor,
    num_history_queries: int,
    normalize_query_lags: bool = False,
) -> torch.Tensor:
    """Original LTE-style auxiliary loss for image-only temporal encoders.

    For each decoder observation state, query one random point in each of
    ``num_history_queries`` bins spanning its entire causal prefix and regress
    the corresponding (detached) ResNet embedding.
    """
    elapsed = observation_indices.to(history_embeddings.dtype).clamp_min(0)
    query_lags = _sample_stratified_history_lags(elapsed, num_history_queries)
    targets = _interpolate_past_embeddings(
        history_embeddings, observation_indices, query_lags
    )
    decoder_query_lags = query_lags
    if normalize_query_lags:
        # Preserve raw episode-relative positions for interpolation, but match
        # the original LTE path's [-1, 1] step_count coordinate for the
        # decoder conditioning input. The history tensors are padded to the
        # dataset-wide maximum episode length.
        max_history_index = max(history_embeddings.shape[1] - 1, 1)
        # Keep the same MIN_MAX normalization expression as the original
        # loader: raw step positions / (dataset_max + 1e-8), then * 2 - 1.
        # The decoder receives a time difference, so the offset cancels.
        decoder_query_lags = query_lags * (2.0 / (max_history_index + 1e-8))
    predictions = decoder(
        observation_latents.unsqueeze(-2).expand(-1, -1, num_history_queries, -1),
        decoder_query_lags.unsqueeze(-1),
    )
    return nn.functional.mse_loss(predictions, targets)


def multi_image_history_reconstruction_loss(
    decoder: MultiImageHistoryDecoder,
    observation_latents: torch.Tensor,
    history_embeddings: Sequence[torch.Tensor],
    observation_indices: torch.Tensor,
    num_history_queries: int,
    normalize_query_lags: bool = False,
) -> torch.Tensor:
    """Average independent per-view reconstruction losses at shared queries."""
    if len(history_embeddings) != len(decoder.image_embedding_dims):
        raise ValueError("number of histories does not match decoder views")
    elapsed = observation_indices.to(history_embeddings[0].dtype).clamp_min(0)
    query_lags = _sample_stratified_history_lags(elapsed, num_history_queries)
    decoder_query_lags = query_lags
    if normalize_query_lags:
        max_history_index = max(history_embeddings[0].shape[1] - 1, 1)
        decoder_query_lags = query_lags * (2.0 / (max_history_index + 1e-8))
    targets = [
        _interpolate_past_embeddings(embeddings, observation_indices, query_lags)
        for embeddings in history_embeddings
    ]
    predictions = decoder(
        observation_latents.unsqueeze(-2).expand(-1, -1, num_history_queries, -1),
        decoder_query_lags.unsqueeze(-1),
    )
    return torch.stack([
        nn.functional.mse_loss(prediction, target)
        for prediction, target in zip(predictions, targets)
    ]).mean()
