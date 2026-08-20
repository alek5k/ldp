"""Regression coverage for LTE-IMG-NoT datasets without ``step_count``."""
import numpy as np
import zarr

from diffusion_policy.dataset.temporal_zarr_image_dataset import (
    TemporalZarrImageDataset,
)


def test_lte_img_not_temporal_prefix_does_not_require_step_count(tmp_path):
    dataset_path = tmp_path / "temporal_images_without_step_count.zarr"
    root = zarr.open(str(dataset_path), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    data.array("action", np.zeros((4, 2), dtype=np.float32))
    data.array("agent_pose", np.zeros((4, 2), dtype=np.float32))
    data.array("full_image", np.zeros((4, 3, 4, 4), dtype=np.float32))
    meta.array("episode_ends", np.array([2, 4], dtype=np.int64))

    dataset = TemporalZarrImageDataset(
        zarr_path=str(dataset_path),
        horizon=2,
        n_obs_steps=2,
        pad_before=0,
        pad_after=0,
        val_ratio=0.0,
        return_temporal_history=True,
    )

    sample = dataset[0]
    assert "step_count" not in dataset.replay_buffer.keys()
    assert sample["temporal_history_mask"].sum().item() == 2
    assert sample["temporal_obs_history_indices"].tolist() == [0, 1]
