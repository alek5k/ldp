"""Shared causal image-state support for LTE image diffusion policies."""
from __future__ import annotations

from typing import Dict

import h5py
import numpy as np
import torch

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.zarr_image_cache import get_zarr_array_cache
from diffusion_policy.model.temporal.image_no_time import (
    ImageHistoryDecoder,
    ImageNoTimeTemporalEncoder,
    MultiImageEmbeddingFusion,
    MultiImageHistoryDecoder,
    gather_history_latents,
    image_history_reconstruction_loss,
    multi_image_history_reconstruction_loss,
)


class LTEImageTemporalMixin:
    """Dataset-prefix and online-state handling shared by LTE decoders.

    A policy supplies ``_encode_temporal_images`` to map the selected RGB
    camera to a feature vector. The recurrent LTE state deliberately remains
    independent of the diffusion decoder architecture.
    """

    @staticmethod
    def _resolve_temporal_rgb_key(shape_meta: dict, configured_key: str | None) -> str:
        keys = [
            key for key, value in shape_meta["obs"].items()
            if value.get("type", "low_dim") == "rgb"
        ]
        if configured_key is not None:
            if configured_key not in keys:
                raise ValueError(f"temporal_rgb_key '{configured_key}' is not an RGB observation")
            return configured_key
        if len(keys) != 1:
            raise ValueError("temporal_rgb_key is required with multiple RGB observations")
        return keys[0]

    @staticmethod
    def _resolve_temporal_rgb_keys(
        shape_meta: dict, configured_key: str | None, configured_keys: list[str] | None,
        use_multi_image_fusion: bool,
    ) -> list[str]:
        available = [
            key for key, value in shape_meta["obs"].items()
            if value.get("type", "low_dim") == "rgb"
        ]
        if configured_keys is not None:
            keys = list(configured_keys)
            if not keys or len(set(keys)) != len(keys) or any(key not in available for key in keys):
                raise ValueError("temporal_rgb_keys must be unique RGB observation keys")
        elif use_multi_image_fusion:
            keys = available
        else:
            keys = [LTEImageTemporalMixin._resolve_temporal_rgb_key(shape_meta, configured_key)]
        if use_multi_image_fusion and len(keys) < 2:
            raise ValueError("multi-image LTE fusion requires at least two temporal_rgb_keys")
        return keys

    def _init_lte_temporal(
        self,
        *,
        shape_meta: dict,
        temporal_rgb_key: str | None,
        temporal_rgb_keys: list[str] | None,
        temporal_multi_image_fusion_enabled: bool,
        temporal_multi_image_fusion_dim: int | None,
        temporal_multi_image_fusion_hidden_dim: int | None,
        temporal_latent_dim: int,
        temporal_hidden_dim: int,
        temporal_recurrent: bool,
        temporal_subsample_frames: int,
        temporal_encode_chunk_size: int,
        temporal_embedding_cache_enabled: bool,
        temporal_embedding_cache_start_epoch: int,
        temporal_embedding_cache_warmup_epochs: int,
        temporal_embedding_cache_refresh_epochs: int,
        temporal_image_zarr_path: str | None,
        temporal_image_hdf5_path: str | None,
        temporal_image_cache_in_memory: bool,
        history_reconstruction: dict | None,
    ) -> None:
        if temporal_subsample_frames < 1:
            raise ValueError("temporal_subsample_frames must be at least one")
        if temporal_embedding_cache_warmup_epochs < 0:
            raise ValueError("temporal_embedding_cache_warmup_epochs must be non-negative")
        if temporal_embedding_cache_start_epoch < 0:
            raise ValueError("temporal_embedding_cache_start_epoch must be non-negative")
        if temporal_embedding_cache_refresh_epochs <= 0:
            raise ValueError("temporal_embedding_cache_refresh_epochs must be positive")
        self.temporal_rgb_keys = self._resolve_temporal_rgb_keys(
            shape_meta, temporal_rgb_key, temporal_rgb_keys,
            temporal_multi_image_fusion_enabled,
        )
        self.temporal_rgb_key = self.temporal_rgb_keys[0]
        self.temporal_multi_image_fusion_enabled = bool(
            temporal_multi_image_fusion_enabled and len(self.temporal_rgb_keys) > 1
        )
        self.temporal_image_embedding_dims = [
            int(self._temporal_image_embedding_dim(key)) for key in self.temporal_rgb_keys
        ]
        self.image_embedding_dim = self.temporal_image_embedding_dims[0]
        self.temporal_image_fusion = None
        if self.temporal_multi_image_fusion_enabled:
            fusion_dim = int(temporal_multi_image_fusion_dim or self.image_embedding_dim)
            fusion_hidden_dim = int(
                temporal_multi_image_fusion_hidden_dim or fusion_dim
            )
            self.temporal_image_fusion = MultiImageEmbeddingFusion(
                self.temporal_image_embedding_dims, fusion_dim, fusion_hidden_dim
            )
            self.image_embedding_dim = fusion_dim
        self.temporal_encoder = ImageNoTimeTemporalEncoder(
            image_embedding_dim=self.image_embedding_dim,
            latent_dim=temporal_latent_dim,
            hidden_dim=temporal_hidden_dim,
            recurrent=temporal_recurrent,
        )
        history_config = history_reconstruction or {}
        self.history_reconstruction_enabled = bool(history_config.get("enabled", True))
        self.history_reconstruction_lambda = float(history_config.get("lambda_history", 1.0))
        self.num_history_queries = int(history_config.get("num_history_queries", 16))
        self.normalize_history_query_lags = bool(
            history_config.get("normalize_query_lags", False)
        )
        if self.num_history_queries <= 0:
            raise ValueError("history_reconstruction.num_history_queries must be positive")
        self.history_decoder = (
            (MultiImageHistoryDecoder(
                latent_dim=temporal_latent_dim,
                image_embedding_dims=self.temporal_image_embedding_dims,
                hidden_dim=int(history_config.get("hidden_dim", temporal_hidden_dim)),
                num_hidden_layers=int(history_config.get("num_hidden_layers", 1)),
            ) if self.temporal_multi_image_fusion_enabled else ImageHistoryDecoder(
                latent_dim=temporal_latent_dim,
                image_embedding_dim=self.image_embedding_dim,
                hidden_dim=int(history_config.get("hidden_dim", temporal_hidden_dim)),
            ))
            if self.history_reconstruction_enabled else None
        )
        self.temporal_subsample_frames = temporal_subsample_frames
        self.temporal_encode_chunk_size = temporal_encode_chunk_size
        self.temporal_embedding_cache_enabled = bool(temporal_embedding_cache_enabled)
        self.temporal_embedding_cache_start_epoch = int(
            temporal_embedding_cache_start_epoch
        )
        self.temporal_embedding_cache_warmup_epochs = int(
            temporal_embedding_cache_warmup_epochs
        )
        self.temporal_embedding_cache_refresh_epochs = int(
            temporal_embedding_cache_refresh_epochs
        )
        self.temporal_image_zarr_path = temporal_image_zarr_path
        self.temporal_image_hdf5_path = temporal_image_hdf5_path
        self.temporal_image_cache_in_memory = bool(temporal_image_cache_in_memory)
        self._temporal_replay_buffer = None
        self._temporal_hdf5_file = None
        self._temporal_hdf5_episode_ends = None
        self._temporal_image_cache = None
        self._temporal_feature_cache = None
        self._temporal_latent_history: list[torch.Tensor] = []
        self._last_loss_components: dict[str, torch.Tensor] = {}

    def get_last_loss_components(self) -> dict[str, torch.Tensor]:
        """Return detached components of the most recent LTE loss call."""
        return dict(self._last_loss_components)

    def _open_temporal_hdf5(self):
        if self.temporal_image_hdf5_path is None:
            raise RuntimeError("temporal_image_hdf5_path is required for HDF5 LTE batches")
        if self._temporal_hdf5_file is None:
            self._temporal_hdf5_file = h5py.File(self.temporal_image_hdf5_path, "r")
            demos = self._temporal_hdf5_file["data"]
            lengths = []
            for index in range(len(demos)):
                demo = demos[f"demo_{index}"]
                action_key = "action" if "action" in demo else "actions"
                lengths.append(int(demo[action_key].shape[0]))
            self._temporal_hdf5_episode_ends = np.cumsum(lengths, dtype=np.int64)
        return self._temporal_hdf5_file

    def _load_hdf5_image_cache(self, key: str) -> np.ndarray:
        """Materialize one selected LTE camera in RAM."""
        if self._temporal_image_cache is None:
            self._temporal_image_cache = {}
        if key not in self._temporal_image_cache:
            file = self._open_temporal_hdf5()
            total_frames = int(self._temporal_hdf5_episode_ends[-1])
            first_images = file[f"data/demo_0/obs/{key}"]
            cache = np.empty((total_frames,) + first_images.shape[1:], dtype=first_images.dtype)
            episode_starts = np.concatenate(([0], self._temporal_hdf5_episode_ends[:-1]))
            for episode_id, episode_start in enumerate(episode_starts):
                images = file[f"data/demo_{episode_id}/obs/{key}"]
                cache[episode_start:self._temporal_hdf5_episode_ends[episode_id]] = images[:]
            gib = cache.nbytes / (1024 ** 3)
            print(f"Cached LTE camera '{key}' in RAM ({gib:.2f} GiB).")
            self._temporal_image_cache[key] = cache
        return self._temporal_image_cache[key]

    def _read_hdf5_history_images(self, indices: torch.Tensor, key: str) -> torch.Tensor:
        file = self._open_temporal_hdf5()
        requested = indices.detach().cpu().numpy().astype(np.int64, copy=False)
        if self.temporal_image_cache_in_memory:
            image = self._load_hdf5_image_cache(key)[requested]
        else:
            unique, inverse = np.unique(requested, return_inverse=True)
            episode_ids = np.searchsorted(self._temporal_hdf5_episode_ends, unique, side="right")
            episode_starts = np.concatenate(([0], self._temporal_hdf5_episode_ends[:-1]))
            image_by_unique = [None] * len(unique)
            for episode_id in np.unique(episode_ids):
                positions = np.flatnonzero(episode_ids == episode_id)
                local_indices = unique[positions] - episode_starts[episode_id]
                images = np.asarray(file[f"data/demo_{episode_id}/obs/{key}"][local_indices])
                for position, image in zip(positions, images):
                    image_by_unique[position] = image
            image = np.asarray(image_by_unique, dtype=np.uint8)[inverse]
        # Preserve compact HWC uint8 data in RAM. CUDA performs the channel
        # move, float cast, and [0,1] normalisation after the smaller transfer.
        image = torch.from_numpy(np.ascontiguousarray(image)).to(self.device)
        return image.permute(0, 3, 1, 2).to(dtype=self.dtype).div_(255.0)

    def _read_history_images(self, indices: torch.Tensor, key: str) -> torch.Tensor:
        if self.temporal_image_hdf5_path is not None:
            return self._read_hdf5_history_images(indices, key)
        if self.temporal_image_zarr_path is None:
            raise RuntimeError("temporal image path is required for LTE training batches")
        if self._temporal_replay_buffer is None:
            self._temporal_replay_buffer = ReplayBuffer.create_from_path(self.temporal_image_zarr_path, mode="r")
        image_array = self._temporal_replay_buffer[key]
        requested = indices.detach().cpu().numpy().astype(np.int64, copy=False)
        if self.temporal_image_cache_in_memory:
            if self._temporal_image_cache is None:
                self._temporal_image_cache = {}
            if key not in self._temporal_image_cache:
                self._temporal_image_cache[key] = get_zarr_array_cache(
                    self.temporal_image_zarr_path, key, self._temporal_replay_buffer
                )
            image = self._temporal_image_cache[key][requested]
        else:
            image = np.asarray(image_array.oindex[requested])
        image = image.astype(np.float32, copy=False)
        if image.size and image.max() > 1.0:
            image = image / 255.0
        return torch.from_numpy(np.ascontiguousarray(image)).to(self.device)

    def _temporal_image_frame_count(self) -> int:
        if self.temporal_image_hdf5_path is not None:
            self._open_temporal_hdf5()
            return int(self._temporal_hdf5_episode_ends[-1])
        if self.temporal_image_zarr_path is None:
            raise RuntimeError("temporal image path is required for embedding caching")
        if self._temporal_replay_buffer is None:
            self._temporal_replay_buffer = ReplayBuffer.create_from_path(
                self.temporal_image_zarr_path, mode="r"
            )
        return int(self._temporal_replay_buffer[self.temporal_rgb_key].shape[0])

    def refresh_temporal_embedding_cache(self, epoch_idx: int) -> bool:
        """Refresh detached LTE visual features when the schedule requires it."""
        if not self.temporal_embedding_cache_enabled:
            return False
        if epoch_idx < self.temporal_embedding_cache_start_epoch:
            return False
        if self._temporal_feature_cache is not None:
            cache_warmup_end_epoch = (
                self.temporal_embedding_cache_start_epoch
                + self.temporal_embedding_cache_warmup_epochs
            )
            # Rebuild on every cached epoch in the warm-up phase.  Returning
            # early here previously made the workspace *report* a refresh
            # while retaining the start-epoch ResNet features.
            if epoch_idx >= cache_warmup_end_epoch and (
                (epoch_idx - cache_warmup_end_epoch)
                % self.temporal_embedding_cache_refresh_epochs
            ) != 0:
                return False
        frame_count = self._temporal_image_frame_count()
        caches = [
            torch.empty((frame_count, dim), device=self.device, dtype=self.dtype)
            for dim in self.temporal_image_embedding_dims
        ]
        with torch.no_grad():
            for start in range(0, frame_count, self.temporal_encode_chunk_size):
                end = min(start + self.temporal_encode_chunk_size, frame_count)
                indices = torch.arange(start, end, dtype=torch.long)
                for cache, key in zip(caches, self.temporal_rgb_keys):
                    cache[start:end] = self._encode_temporal_images(
                        self._read_history_images(indices, key).to(dtype=self.dtype), key
                    )
        self._temporal_feature_cache = caches
        return True

    def _history_latents_and_embeddings_from_batch(
        self, batch: dict
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        try:
            indices_cpu = batch["temporal_history_image_indices"]
            valid_mask_cpu = batch["temporal_history_mask"].bool()
            observation_indices = batch["temporal_obs_history_indices"].to(
                device=self.device, dtype=torch.long
            )
        except KeyError as exc:
            raise KeyError("LTE batches require return_temporal_history=true") from exc
        if indices_cpu.device.type != "cpu" or valid_mask_cpu.device.type != "cpu":
            raise ValueError(
                "LTE history image indices and mask must remain on CPU for cache lookup"
            )
        batch_size, history_len = indices_cpu.shape
        embedding_views = [
            torch.zeros(batch_size, history_len, dim, device=self.device, dtype=self.dtype)
            for dim in self.temporal_image_embedding_dims
        ]
        valid_indices = indices_cpu[valid_mask_cpu]
        valid_mask = valid_mask_cpu.to(device=self.device, dtype=torch.bool)
        if self._temporal_feature_cache is not None:
            for embeddings, cache in zip(embedding_views, self._temporal_feature_cache):
                embeddings[valid_mask] = cache[valid_indices.to(device=self.device, dtype=torch.long)]
        else:
            # Match LTE-IMG-NoT: temporal recurrence does not backpropagate
            # through the entire image prefix into the visual encoder.
            with torch.no_grad():
                for embeddings, key in zip(embedding_views, self.temporal_rgb_keys):
                    encoded = []
                    for start in range(0, valid_indices.numel(), self.temporal_encode_chunk_size):
                        images = self._read_history_images(
                            valid_indices[start:start + self.temporal_encode_chunk_size], key
                        )
                        encoded.append(self._encode_temporal_images(images.to(dtype=self.dtype), key))
                    if encoded:
                        embeddings[valid_mask] = torch.cat(encoded, dim=0)
        encoder_input = (
            self.temporal_image_fusion(embedding_views)
            if self.temporal_image_fusion is not None else embedding_views[0]
        )
        states = self.temporal_encoder.encode_history(encoder_input, valid_mask)
        return gather_history_latents(states, observation_indices), embedding_views, observation_indices

    def _history_latents_from_batch(self, batch: dict) -> torch.Tensor:
        return self._history_latents_and_embeddings_from_batch(batch)[0]

    def _history_reconstruction_loss(
        self,
        observation_latents: torch.Tensor,
        history_embeddings: list[torch.Tensor],
        observation_indices: torch.Tensor,
    ) -> torch.Tensor:
        if self.history_decoder is None:
            return torch.zeros((), device=observation_latents.device, dtype=observation_latents.dtype)
        if self.temporal_image_fusion is not None:
            return multi_image_history_reconstruction_loss(
                self.history_decoder, observation_latents, history_embeddings,
                observation_indices, self.num_history_queries,
                normalize_query_lags=self.normalize_history_query_lags,
            )
        return image_history_reconstruction_loss(
            self.history_decoder, observation_latents, history_embeddings[0],
            observation_indices, self.num_history_queries,
            normalize_query_lags=self.normalize_history_query_lags,
        )

    def _fallback_temporal_latents(self, nobs: Dict[str, torch.Tensor]) -> torch.Tensor:
        images = nobs[self.temporal_rgb_key][:, :self.n_obs_steps]
        batch_size, steps = images.shape[:2]
        embeddings = [
            self._encode_temporal_images(
                nobs[key][:, :steps].reshape(-1, *nobs[key].shape[2:]), key
            ).reshape(batch_size, steps, -1)
            for key in self.temporal_rgb_keys
        ]
        encoder_input = self.temporal_image_fusion(embeddings) if self.temporal_image_fusion else embeddings[0]
        return self.temporal_encoder.encode_history(encoder_input)

    def _online_temporal_latents(self, nobs: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = nobs[self.temporal_rgb_key].shape[0]
        if not self._temporal_latent_history:
            return self._fallback_temporal_latents(nobs)
        latest = self._temporal_latent_history[-1]
        if latest.shape[0] != batch_size:
            return self._fallback_temporal_latents(nobs)
        required = self.n_obs_steps * self.temporal_subsample_frames
        states = self._temporal_latent_history
        if len(states) < required:
            states = [states[0]] * (required - len(states)) + states
        start = len(states) - required + (self.temporal_subsample_frames - 1)
        return torch.stack(states[start::self.temporal_subsample_frames], dim=1)

    def reset(self):
        self._temporal_latent_history = []

    @torch.no_grad()
    def advance_temporal_state(self, image: torch.Tensor):
        if self.temporal_image_fusion is not None:
            raise ValueError(
                "multi-image LTE requires advance_temporal_state_from_observation"
            )
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError("image must have shape (B, C, H, W)")
        embedding = self._encode_temporal_images(image.to(device=self.device, dtype=self.dtype))
        previous = self._temporal_latent_history[-1] if self._temporal_latent_history else None
        self._temporal_latent_history.append(self.temporal_encoder(embedding, previous))

    @torch.no_grad()
    def advance_temporal_state_from_observation(self, observation: Dict[str, torch.Tensor]):
        embeddings = []
        for key in self.temporal_rgb_keys:
            image = observation[key]
            if image.ndim == 5:
                image = image[:, -1]
            if image.ndim != 4:
                raise ValueError("temporal observation images must have shape (B, C, H, W)")
            embeddings.append(self._encode_temporal_images(
                image.to(device=self.device, dtype=self.dtype), key
            ))
        embedding = self.temporal_image_fusion(embeddings) if self.temporal_image_fusion else embeddings[0]
        previous = self._temporal_latent_history[-1] if self._temporal_latent_history else None
        self._temporal_latent_history.append(self.temporal_encoder(embedding, previous))
