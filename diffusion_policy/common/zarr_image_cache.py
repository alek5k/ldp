"""Process-local RAM cache for temporal Zarr image arrays.

The temporal image dataset and LTE policy run in the same training process.
Sharing this cache lets the normal observation path and the causal-prefix path
use one decompressed array, matching the original eager dataset loader.
"""
from __future__ import annotations

import os

import numpy as np

from diffusion_policy.common.replay_buffer import ReplayBuffer


_ZARR_ARRAY_CACHE: dict[tuple[str, str], np.ndarray] = {}


def get_zarr_array_cache(
    zarr_path: str,
    key: str,
    replay_buffer: ReplayBuffer | None = None,
) -> np.ndarray:
    """Return one eagerly materialized Zarr data array for this process."""
    cache_key = (os.path.realpath(os.path.expanduser(zarr_path)), key)
    cached = _ZARR_ARRAY_CACHE.get(cache_key)
    if cached is None:
        if replay_buffer is None:
            replay_buffer = ReplayBuffer.create_from_path(cache_key[0], mode="r")
        cached = np.asarray(replay_buffer[key][:])
        _ZARR_ARRAY_CACHE[cache_key] = cached
        print(
            f"Cached temporal Zarr '{key}' in RAM "
            f"({cached.nbytes / 1024 ** 3:.2f} GiB)."
        )
    return cached
