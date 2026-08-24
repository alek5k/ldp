"""Diffusion U-Net policy with the LTE-IMG-NoT conditioning feature.

The ordinary visual observation path remains end-to-end trainable.  LTE sees
the same ResNet feature, but detached, matching the original implementation:
the auxiliary causal state must not backpropagate through a whole episode of
images into the visual encoder.
"""
from __future__ import annotations

from typing import Dict

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.zarr_image_cache import get_zarr_array_cache
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.temporal.image_no_time import (
    ImageHistoryDecoder,
    ImageNoTimeTemporalEncoder,
    gather_history_latents,
    image_history_reconstruction_loss,
)
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class DiffusionUnetLTEImagePolicy(BaseImagePolicy):
    """LDP-native implementation of LTE-IMG-NoT.

    ``temporal_image_zarr_path`` is used only during optimization, when each
    batch supplies lightweight absolute image indices for complete episode
    prefixes.  At rollout, :meth:`advance_temporal_state` updates the state
    after every environment observation.
    """

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        obs_encoder: MultiImageObsEncoder,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        temporal_image_zarr_path: str | None = None,
        temporal_image_hdf5_path: str | None = None,
        temporal_image_cache_in_memory: bool = False,
        temporal_rgb_key: str | None = None,
        temporal_latent_dim: int = 32,
        temporal_hidden_dim: int = 128,
        temporal_num_hidden_layers: int = 1,
        temporal_recurrent: bool = True,
        temporal_subsample_frames: int = 1,
        temporal_encode_chunk_size: int = 256,
        temporal_embedding_cache_enabled: bool = False,
        temporal_embedding_cache_start_epoch: int = 5,
        temporal_embedding_cache_warmup_epochs: int = 20,
        temporal_embedding_cache_refresh_epochs: int = 5,
        history_reconstruction: dict | None = None,
        num_inference_steps: int | None = None,
        obs_as_global_cond: bool = True,
        diffusion_step_embed_dim: int = 256,
        down_dims=(256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        **kwargs,
    ):
        super().__init__()
        action_shape = shape_meta["action"]["shape"]
        if len(action_shape) != 1:
            raise ValueError("LTE image policy requires a vector action space")
        if temporal_subsample_frames < 1:
            raise ValueError("temporal_subsample_frames must be at least one")
        if temporal_num_hidden_layers < 1:
            raise ValueError("temporal_num_hidden_layers must be at least one")
        if temporal_embedding_cache_warmup_epochs < 0:
            raise ValueError("temporal_embedding_cache_warmup_epochs must be non-negative")
        if temporal_embedding_cache_start_epoch < 0:
            raise ValueError("temporal_embedding_cache_start_epoch must be non-negative")
        if temporal_embedding_cache_refresh_epochs <= 0:
            raise ValueError("temporal_embedding_cache_refresh_epochs must be positive")
        self.obs_encoder = obs_encoder
        self.temporal_rgb_key = self._resolve_temporal_rgb_key(
            shape_meta, temporal_rgb_key
        )
        self.image_embedding_dim = self._rgb_feature_dim()
        self.temporal_encoder = ImageNoTimeTemporalEncoder(
            image_embedding_dim=self.image_embedding_dim,
            latent_dim=temporal_latent_dim,
            hidden_dim=temporal_hidden_dim,
            recurrent=temporal_recurrent,
            num_hidden_layers=temporal_num_hidden_layers,
        )
        history_config = history_reconstruction or {}
        self.history_reconstruction_enabled = bool(history_config.get("enabled", True))
        self.history_reconstruction_lambda = float(history_config.get("lambda_history", 1.0))
        self.num_history_queries = int(history_config.get("num_history_queries", 16))
        self.history_decoder_num_hidden_layers = int(
            history_config.get("num_hidden_layers", 1)
        )
        self.normalize_history_query_lags = bool(
            history_config.get("normalize_query_lags", False)
        )
        if self.num_history_queries <= 0:
            raise ValueError("history_reconstruction.num_history_queries must be positive")
        if self.history_decoder_num_hidden_layers < 1:
            raise ValueError(
                "history_reconstruction.num_hidden_layers must be at least one"
            )
        self.history_decoder = (
            ImageHistoryDecoder(
                latent_dim=temporal_latent_dim,
                image_embedding_dim=self.image_embedding_dim,
                hidden_dim=int(history_config.get("hidden_dim", temporal_hidden_dim)),
                num_hidden_layers=self.history_decoder_num_hidden_layers,
            )
            if self.history_reconstruction_enabled else None
        )

        self.action_dim = action_shape[0]
        self.obs_feature_dim = obs_encoder.output_shape()[0]
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
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

        condition_feature_dim = self.obs_feature_dim + temporal_latent_dim
        input_dim = self.action_dim + condition_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = self.action_dim
            global_cond_dim = condition_feature_dim * n_obs_steps
        self.model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=self.action_dim,
            obs_dim=0 if obs_as_global_cond else condition_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.kwargs = kwargs
        self.num_inference_steps = (
            noise_scheduler.config.num_train_timesteps
            if num_inference_steps is None else num_inference_steps
        )

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
            raise ValueError(
                "temporal_rgb_key is required when an environment has multiple RGB observations"
            )
        return keys[0]

    def _rgb_model(self):
        return (
            self.obs_encoder.key_model_map["rgb"]
            if self.obs_encoder.share_rgb_model
            else self.obs_encoder.key_model_map[self.temporal_rgb_key]
        )

    def _rgb_feature_dim(self) -> int:
        shape = self.obs_encoder.key_shape_map[self.temporal_rgb_key]
        example = torch.zeros((1,) + shape, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            return int(self._rgb_model()(self.obs_encoder.key_transform_map[self.temporal_rgb_key](example)).shape[-1])

    def _encode_temporal_images(self, images: torch.Tensor) -> torch.Tensor:
        """Return the ResNet feature used as LTE's image-only input."""
        if images.ndim != 4:
            raise ValueError("temporal images must have shape (B, C, H, W)")
        image = self.obs_encoder.key_transform_map[self.temporal_rgb_key](images)
        return self._rgb_model()(image)

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
        if self.temporal_image_cache_in_memory:
            image = self._load_hdf5_image_cache()[requested]
        else:
            unique, inverse = np.unique(requested, return_inverse=True)
            episode_ids = np.searchsorted(self._temporal_hdf5_episode_ends, unique, side="right")
            episode_starts = np.concatenate(([0], self._temporal_hdf5_episode_ends[:-1]))
            image_by_unique = [None] * len(unique)
            for episode_id in np.unique(episode_ids):
                positions = np.flatnonzero(episode_ids == episode_id)
                local_indices = unique[positions] - episode_starts[episode_id]
                images = np.asarray(
                    file[f"data/demo_{episode_id}/obs/{self.temporal_rgb_key}"][local_indices]
                )
                for position, image in zip(positions, images):
                    image_by_unique[position] = image
            image = np.asarray(image_by_unique, dtype=np.uint8)[inverse]
        # Keep the host cache in compact HWC uint8 form. The old path expanded
        # every sampled prefix frame to float32 on CPU before sending it to
        # CUDA. Move that layout conversion and normalisation onto the GPU.
        image = torch.from_numpy(np.ascontiguousarray(image)).to(self.device)
        return image.permute(0, 3, 1, 2).to(dtype=self.dtype).div_(255.0)

    def _load_hdf5_image_cache(self) -> np.ndarray:
        """Materialize the one LTE camera in RAM, matching the original LTE loader."""
        if self._temporal_image_cache is None:
            file = self._open_temporal_hdf5()
            total_frames = int(self._temporal_hdf5_episode_ends[-1])
            first_images = file[f"data/demo_0/obs/{self.temporal_rgb_key}"]
            cache = np.empty((total_frames,) + first_images.shape[1:], dtype=first_images.dtype)
            episode_starts = np.concatenate(([0], self._temporal_hdf5_episode_ends[:-1]))
            for episode_id, episode_start in enumerate(episode_starts):
                images = file[f"data/demo_{episode_id}/obs/{self.temporal_rgb_key}"]
                cache[episode_start:self._temporal_hdf5_episode_ends[episode_id]] = images[:]
            self._temporal_image_cache = cache
            gib = cache.nbytes / (1024 ** 3)
            print(f"Cached LTE camera '{self.temporal_rgb_key}' in RAM ({gib:.2f} GiB).")
        return self._temporal_image_cache

    def _read_history_images(self, indices: torch.Tensor) -> torch.Tensor:
        if self.temporal_image_hdf5_path is not None:
            return self._read_hdf5_history_images(indices)
        if self.temporal_image_zarr_path is None:
            raise RuntimeError(
                "temporal_image_zarr_path is required for LTE training batches"
            )
        if self._temporal_replay_buffer is None:
            self._temporal_replay_buffer = ReplayBuffer.create_from_path(
                self.temporal_image_zarr_path, mode="r"
            )
        image_array = self._temporal_replay_buffer["full_image"]
        index_array = indices.detach().cpu().numpy().astype(np.int64, copy=False)
        if self.temporal_image_cache_in_memory:
            self._temporal_image_cache = get_zarr_array_cache(
                self.temporal_image_zarr_path, "full_image", self._temporal_replay_buffer
            )
            image = self._temporal_image_cache[index_array]
        else:
            image = np.asarray(image_array.oindex[index_array])
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
        return int(self._temporal_replay_buffer["full_image"].shape[0])

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
        cache = torch.empty(
            (frame_count, self.image_embedding_dim),
            device=self.device,
            dtype=self.dtype,
        )
        with torch.no_grad():
            for start in range(0, frame_count, self.temporal_encode_chunk_size):
                end = min(start + self.temporal_encode_chunk_size, frame_count)
                indices = torch.arange(start, end, dtype=torch.long)
                cache[start:end] = self._encode_temporal_images(
                    self._read_history_images(indices).to(dtype=self.dtype)
                )
        self._temporal_feature_cache = cache
        return True

    def _history_latents_and_embeddings_from_batch(
        self, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            indices_cpu = batch["temporal_history_image_indices"]
            valid_mask_cpu = batch["temporal_history_mask"].bool()
            observation_indices = batch["temporal_obs_history_indices"].to(
                device=self.device, dtype=torch.long
            )
        except KeyError as exc:
            raise KeyError(
                "LTE batches require a dataset with return_temporal_history=true"
            ) from exc
        if indices_cpu.device.type != "cpu" or valid_mask_cpu.device.type != "cpu":
            raise ValueError(
                "LTE history image indices and mask must remain on CPU for cache lookup"
            )
        batch_size, history_len = indices_cpu.shape
        valid_indices = indices_cpu[valid_mask_cpu]
        valid_mask = valid_mask_cpu.to(device=self.device, dtype=torch.bool)
        embeddings = torch.zeros(
            batch_size, history_len, self.image_embedding_dim,
            device=self.device, dtype=self.dtype,
        )
        if self._temporal_feature_cache is not None:
            embeddings[valid_mask] = self._temporal_feature_cache[
                valid_indices.to(device=self.device, dtype=torch.long)
            ]
        else:
            # Detach this branch exactly as in the source LTE implementation.
            with torch.no_grad():
                encoded = []
                for start in range(0, valid_indices.numel(), self.temporal_encode_chunk_size):
                    images = self._read_history_images(
                        valid_indices[start:start + self.temporal_encode_chunk_size]
                    ).to(dtype=self.dtype)
                    encoded.append(self._encode_temporal_images(images))
                if encoded:
                    embeddings[valid_mask] = torch.cat(encoded, dim=0)
        states = self.temporal_encoder.encode_history(embeddings, valid_mask)
        return gather_history_latents(states, observation_indices), embeddings, observation_indices

    def _history_latents_from_batch(self, batch: dict) -> torch.Tensor:
        return self._history_latents_and_embeddings_from_batch(batch)[0]

    def _history_reconstruction_loss(
        self,
        observation_latents: torch.Tensor,
        history_embeddings: torch.Tensor,
        observation_indices: torch.Tensor,
    ) -> torch.Tensor:
        if self.history_decoder is None:
            return torch.zeros((), device=observation_latents.device, dtype=observation_latents.dtype)
        return image_history_reconstruction_loss(
            self.history_decoder,
            observation_latents,
            history_embeddings,
            observation_indices,
            self.num_history_queries,
            normalize_query_lags=self.normalize_history_query_lags,
        )

    def _fallback_temporal_latents(self, nobs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Use the supplied window only when no rollout state is available.

        This is useful for offline sample visualisation.  Real evaluations use
        ``advance_temporal_state`` and therefore retain the full prefix.
        """
        images = nobs[self.temporal_rgb_key][:, :self.n_obs_steps]
        batch_size, steps = images.shape[:2]
        embeddings = self._encode_temporal_images(images.reshape(-1, *images.shape[2:]))
        embeddings = embeddings.reshape(batch_size, steps, -1)
        return self.temporal_encoder.encode_history(embeddings)

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
        """Advance LTE once per real environment observation.

        The temporal runner calls this after reset and after each executed
        action, including actions between diffusion replans.
        """
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError("image must have shape (B, C, H, W)")
        image = image.to(device=self.device, dtype=self.dtype)
        embedding = self._encode_temporal_images(image)
        previous = self._temporal_latent_history[-1] if self._temporal_latent_history else None
        self._temporal_latent_history.append(self.temporal_encoder(embedding, previous))

    @torch.no_grad()
    def advance_temporal_state_from_observation(self, observation: Dict[str, torch.Tensor]):
        """Advance from a runner observation dictionary, using the LTE camera."""
        image = observation[self.temporal_rgb_key]
        if image.ndim == 5:
            image = image[:, -1]
        self.advance_temporal_state(image)

    def conditional_sample(self, condition_data, condition_mask,
                           local_cond=None, global_cond=None, generator=None, **kwargs):
        trajectory = torch.randn(
            condition_data.shape, dtype=condition_data.dtype,
            device=condition_data.device, generator=generator,
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for timestep in self.noise_scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = self.model(
                trajectory, timestep, local_cond=local_cond, global_cond=global_cond
            )
            trajectory = self.noise_scheduler.step(
                model_output, timestep, trajectory, generator=generator, **kwargs
            ).prev_sample
        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def _condition_features(self, nobs: Dict[str, torch.Tensor], temporal_latents: torch.Tensor):
        batch_size = temporal_latents.shape[0]
        tobs = temporal_latents.shape[1]
        obs_input = dict_apply(
            nobs, lambda value: value[:, :tobs].reshape(-1, *value.shape[2:])
        )
        obs_features = self.obs_encoder(obs_input).reshape(batch_size, tobs, -1)
        return torch.cat([obs_features, temporal_latents], dim=-1)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        batch_size = nobs[self.temporal_rgb_key].shape[0]
        temporal_latents = self._online_temporal_latents(nobs)
        condition_features = self._condition_features(nobs, temporal_latents)
        if self.obs_as_global_cond:
            global_cond = condition_features.reshape(batch_size, -1)
            cond_data = torch.zeros(
                (batch_size, self.horizon, self.action_dim), device=self.device, dtype=self.dtype
            )
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            local_cond = None
        else:
            cond_data = torch.zeros(
                (batch_size, self.horizon, self.action_dim + condition_features.shape[-1]),
                device=self.device, dtype=self.dtype,
            )
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :self.n_obs_steps, self.action_dim:] = condition_features
            cond_mask[:, :self.n_obs_steps, self.action_dim:] = True
            local_cond = global_cond = None
        sample = self.conditional_sample(
            cond_data, cond_mask, local_cond=local_cond, global_cond=global_cond, **self.kwargs
        )
        action_pred = self.normalizer["action"].unnormalize(sample[..., :self.action_dim])
        start = self.n_obs_steps - 1
        return {"action": action_pred[:, start:start + self.n_action_steps], "action_pred": action_pred}

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_last_loss_components(self) -> dict[str, torch.Tensor]:
        """Return detached components of the most recent LTE loss call."""
        return dict(self._last_loss_components)

    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch["obs"])
        actions = self.normalizer["action"].normalize(batch["action"])
        batch_size, horizon = actions.shape[:2]
        temporal_latents, history_embeddings, observation_indices = (
            self._history_latents_and_embeddings_from_batch(batch)
        )
        condition_features = self._condition_features(nobs, temporal_latents)
        trajectory = actions
        if self.obs_as_global_cond:
            global_cond = condition_features.reshape(batch_size, -1)
            local_cond = None
            condition_data = trajectory
        else:
            local_cond = global_cond = None
            condition_data = torch.cat([actions, condition_features], dim=-1)
            trajectory = condition_data.detach()
        condition_mask = self.mask_generator(trajectory.shape)
        noise = torch.randn_like(trajectory)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        noisy_trajectory[condition_mask] = condition_data[condition_mask]
        pred = self.model(
            noisy_trajectory, timesteps, local_cond=local_cond, global_cond=global_cond
        )
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "sample":
            target = trajectory
        else:
            raise ValueError("Unsupported diffusion prediction type")
        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * (~condition_mask).type(loss.dtype)
        diffusion_loss = reduce(loss, "b ... -> b (...)", "mean").mean()
        reconstruction_loss = self._history_reconstruction_loss(
            temporal_latents, history_embeddings, observation_indices
        )
        total_loss = (
            diffusion_loss
            + self.history_reconstruction_lambda * reconstruction_loss
        )
        self._last_loss_components = {
            "total": total_loss.detach(),
            "diffusion": diffusion_loss.detach(),
            "history": reconstruction_loss.detach(),
        }
        return total_loss
