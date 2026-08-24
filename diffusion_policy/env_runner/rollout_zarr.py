"""Utilities shared by image runners when exporting policy rollouts."""

from pathlib import Path
from typing import Iterable, Mapping, Optional

from diffusion_policy.common.replay_buffer import ReplayBuffer


def create_rollout_replay_buffer(path: str, mode: str) -> ReplayBuffer:
    if mode not in {"w", "a"}:
        raise ValueError("zarr_mode must be 'w' (new store) or 'a' (append)")
    zarr_path = Path(path)
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    return ReplayBuffer.create_from_path(str(zarr_path), mode=mode)


def append_recorded_episodes(
    replay_buffer: ReplayBuffer,
    episodes: Iterable[Optional[Mapping]],
) -> None:
    for episode in episodes:
        if episode is not None:
            replay_buffer.add_episode(dict(episode), compressors="disk")
