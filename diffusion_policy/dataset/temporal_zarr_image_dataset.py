"""Dataset adapter for the Zarr episodes recorded by the temporal tasks.

The adapter deliberately preserves the temporal-analysis fields in the source
store.  Training consumes only ``full_image``, ``agent_pose`` and ``action``;
the remaining arrays stay available to analysis tools in the same Zarr file.
"""
from typing import Dict
import copy

import numpy as np
import torch

from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer


class TemporalZarrImageDataset(BaseImageDataset):
    """Read TemporalDiffusionPolicy-style recordings without conversion.

    Images are stored channel-first as either float images in [0, 1] (the
    native recorder format) or uint8 images.  The policy-facing observation
    names are standardised to ``image`` and ``agent_pose``.  LTE-IMG-NoT does
    not require a recorded ``step_count`` field: causal prefixes come directly
    from ``meta/episode_ends``.
    """

    def __init__(
        self,
        zarr_path: str,
        horizon: int,
        pad_before: int = 0,
        pad_after: int = 0,
        n_obs_steps: int = None,
        subsample_frames: int = 1,
        seed: int = 42,
        val_ratio: float = 0.02,
        max_train_episodes=None,
        return_temporal_history: bool = False,
    ):
        super().__init__()
        # Keep the (large) image arrays in their on-disk Zarr store.  Copying
        # a full temporal dataset into RAM delays startup by minutes and is
        # unnecessary because SequenceSampler reads only one sequence at once.
        self.replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode="r")
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        if max_train_episodes is not None:
            valid_indices = np.flatnonzero(train_mask)
            train_mask[valid_indices[int(max_train_episodes):]] = False

        if n_obs_steps is None:
            n_obs_steps = horizon
        if n_obs_steps > horizon:
            raise ValueError("n_obs_steps cannot exceed horizon")
        if subsample_frames < 1:
            raise ValueError("subsample_frames must be at least one")

        self.n_obs_steps = n_obs_steps
        self.subsample_frames = subsample_frames
        self.sequence_length = (
            (horizon - n_obs_steps) + n_obs_steps * subsample_frames)
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.sequence_length,
            pad_before=pad_before,
            pad_after=pad_after,
            # Images are fetched sparsely below: with 20x frame spacing the
            # generic sampler would otherwise decompress all 400 intervening
            # frames merely to retain 20 of them.
            keys=["agent_pose", "action"],
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.return_temporal_history = return_temporal_history
        episode_ends = np.asarray(self.replay_buffer.episode_ends[:], dtype=np.int64)
        self.temporal_episode_ends = episode_ends
        self.temporal_episode_starts = np.concatenate(([0], episode_ends[:-1]))
        self.max_episode_length = int(
            np.max(self.temporal_episode_ends - self.temporal_episode_starts)
        )

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.sequence_length,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            keys=["agent_pose", "action"],
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer.fit(
            data={
                "action": self.replay_buffer["action"][:],
                "agent_pose": self.replay_buffer["agent_pose"][:],
            },
            last_n_dims=1,
            mode=mode,
            **kwargs,
        )
        normalizer["image"] = get_image_range_normalizer()
        return normalizer

    def __len__(self):
        return len(self.sampler)

    @staticmethod
    def _image_to_float(image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32, copy=False)
        if image.size and image.max() > 1.0:
            image = image / 255.0
        return np.ascontiguousarray(image)

    def _sample_images(self, idx: int, sequence_indices: np.ndarray) -> np.ndarray:
        """Read only selected image frames while matching sampler padding."""
        buffer_start, buffer_end, sample_start, sample_end = self.sampler.indices[idx]
        clipped = np.clip(sequence_indices, sample_start, sample_end - 1)
        source_indices = buffer_start + clipped - sample_start
        image_array = self.replay_buffer["full_image"]
        return np.asarray(image_array.oindex[source_indices])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        past_length = self.n_obs_steps * self.subsample_frames
        observation_indices = np.arange(
            self.subsample_frames - 1, past_length, self.subsample_frames)
        action = np.concatenate([
            sample["action"][:past_length][observation_indices],
            sample["action"][past_length:],
        ])
        data = {
            "obs": {
                "image": self._image_to_float(
                    self._sample_images(idx, observation_indices)),
                "agent_pose": sample["agent_pose"][observation_indices].astype(np.float32),
            },
            "action": action.astype(np.float32),
        }
        if self.return_temporal_history:
            # A recurrent temporal encoder must start at the genuine episode
            # boundary, rather than at the left edge of this sampled window.
            # Keep absolute image indices lightweight; the policy reads and
            # encodes them in bounded chunks in the training process.
            buffer_start, _, sample_start, sample_end = self.sampler.indices[idx]
            source_indices = buffer_start + np.clip(
                observation_indices, sample_start, sample_end - 1
            ) - sample_start
            last_observation = int(source_indices[-1])
            episode_id = int(np.searchsorted(
                self.temporal_episode_ends, last_observation, side="right"
            ))
            episode_start = int(self.temporal_episode_starts[episode_id])
            history_length = last_observation - episode_start + 1
            history_indices = np.zeros(self.max_episode_length, dtype=np.int64)
            history_indices[:history_length] = np.arange(
                episode_start, last_observation + 1, dtype=np.int64
            )
            data["temporal_history_image_indices"] = history_indices
            data["temporal_history_mask"] = (
                np.arange(self.max_episode_length) < history_length
            )
            data["temporal_obs_history_indices"] = (
                source_indices - episode_start
            ).astype(np.int64)
        return dict_apply(data, torch.from_numpy)
