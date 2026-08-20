"""Shared causal image-state support for LTE image diffusion policies."""
from __future__ import annotations

from typing import Dict

import h5py
import numpy as np
import torch

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.model.temporal.image_no_time import (
    ImageNoTimeTemporalEncoder,
    gather_history_latents,
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

    def _init_lte_temporal(
        self,
        *,
        shape_meta: dict,
        temporal_rgb_key: str | None,
        temporal_latent_dim: int,
        temporal_hidden_dim: int,
        temporal_recurrent: bool,
        temporal_subsample_frames: int,
        temporal_encode_chunk_size: int,
        temporal_image_zarr_path: str | None,
        temporal_image_hdf5_path: str | None,
    ) -> None:
        if temporal_subsample_frames < 1:
            raise ValueError("temporal_subsample_frames must be at least one")
        self.temporal_rgb_key = self._resolve_temporal_rgb_key(shape_meta, temporal_rgb_key)
        self.image_embedding_dim = int(self._temporal_image_embedding_dim())
        self.temporal_encoder = ImageNoTimeTemporalEncoder(
            image_embedding_dim=self.image_embedding_dim,
            latent_dim=temporal_latent_dim,
            hidden_dim=temporal_hidden_dim,
            recurrent=temporal_recurrent,
        )
        self.temporal_subsample_frames = temporal_subsample_frames
        self.temporal_encode_chunk_size = temporal_encode_chunk_size
        self.temporal_image_zarr_path = temporal_image_zarr_path
        self.temporal_image_hdf5_path = temporal_image_hdf5_path
        self._temporal_replay_buffer = None
        self._temporal_hdf5_file = None
        self._temporal_hdf5_episode_ends = None
        self._temporal_latent_history: list[torch.Tensor] = []

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

    def _read_hdf5_history_images(self, indices: torch.Tensor) -> torch.Tensor:
        file = self._open_temporal_hdf5()
        requested = indices.detach().cpu().numpy().astype(np.int64, copy=False)
        unique, inverse = np.unique(requested, return_inverse=True)
        episode_ids = np.searchsorted(self._temporal_hdf5_episode_ends, unique, side="right")
        episode_starts = np.concatenate(([0], self._temporal_hdf5_episode_ends[:-1]))
        image_by_unique = [None] * len(unique)
        for episode_id in np.unique(episode_ids):
            positions = np.flatnonzero(episode_ids == episode_id)
            local_indices = unique[positions] - episode_starts[episode_id]
            images = np.asarray(file[f"data/demo_{episode_id}/obs/{self.temporal_rgb_key}"][local_indices])
            for position, image in zip(positions, images):
                image_by_unique[position] = image
        image = np.asarray(image_by_unique, dtype=np.uint8)[inverse]
        image = np.moveaxis(image, -1, 1).astype(np.float32) / 255.0
        return torch.from_numpy(np.ascontiguousarray(image)).to(self.device)

    def _read_history_images(self, indices: torch.Tensor) -> torch.Tensor:
        if self.temporal_image_hdf5_path is not None:
            return self._read_hdf5_history_images(indices)
        if self.temporal_image_zarr_path is None:
            raise RuntimeError("temporal image path is required for LTE training batches")
        if self._temporal_replay_buffer is None:
            self._temporal_replay_buffer = ReplayBuffer.create_from_path(self.temporal_image_zarr_path, mode="r")
        image_array = self._temporal_replay_buffer["full_image"]
        image = np.asarray(image_array.oindex[indices.detach().cpu().numpy().astype(np.int64, copy=False)])
        image = image.astype(np.float32, copy=False)
        if image.size and image.max() > 1.0:
            image = image / 255.0
        return torch.from_numpy(np.ascontiguousarray(image)).to(self.device)

    def _history_latents_from_batch(self, batch: dict) -> torch.Tensor:
        try:
            indices = batch["temporal_history_image_indices"]
            valid_mask = batch["temporal_history_mask"].bool()
            observation_indices = batch["temporal_obs_history_indices"]
        except KeyError as exc:
            raise KeyError("LTE batches require return_temporal_history=true") from exc
        batch_size, history_len = indices.shape
        embeddings = torch.zeros(batch_size, history_len, self.image_embedding_dim,
                                 device=self.device, dtype=self.dtype)
        valid_indices = indices[valid_mask]
        # Match LTE-IMG-NoT: temporal recurrence does not backpropagate through
        # the entire image prefix into the visual encoder.
        with torch.no_grad():
            encoded = []
            for start in range(0, valid_indices.numel(), self.temporal_encode_chunk_size):
                images = self._read_history_images(valid_indices[start:start + self.temporal_encode_chunk_size])
                encoded.append(self._encode_temporal_images(images.to(dtype=self.dtype)))
            if encoded:
                embeddings[valid_mask] = torch.cat(encoded, dim=0)
        states = self.temporal_encoder.encode_history(embeddings, valid_mask)
        return gather_history_latents(states, observation_indices)

    def _fallback_temporal_latents(self, nobs: Dict[str, torch.Tensor]) -> torch.Tensor:
        images = nobs[self.temporal_rgb_key][:, :self.n_obs_steps]
        batch_size, steps = images.shape[:2]
        embeddings = self._encode_temporal_images(images.reshape(-1, *images.shape[2:]))
        return self.temporal_encoder.encode_history(embeddings.reshape(batch_size, steps, -1))

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
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError("image must have shape (B, C, H, W)")
        embedding = self._encode_temporal_images(image.to(device=self.device, dtype=self.dtype))
        previous = self._temporal_latent_history[-1] if self._temporal_latent_history else None
        self._temporal_latent_history.append(self.temporal_encoder(embedding, previous))

    @torch.no_grad()
    def advance_temporal_state_from_observation(self, observation: Dict[str, torch.Tensor]):
        image = observation[self.temporal_rgb_key]
        if image.ndim == 5:
            image = image[:, -1]
        self.advance_temporal_state(image)
