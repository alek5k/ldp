"""Regression coverage for LTE-IMG-NoT datasets without ``step_count``."""
import numpy as np
import torch
import zarr

from diffusion_policy.dataset.temporal_zarr_image_dataset import (
    TemporalZarrImageDataset,
)
from diffusion_policy.model.temporal.image_no_time import (
    ImageHistoryDecoder,
    ImageNoTimeTemporalEncoder,
    MultiImageEmbeddingFusion,
    MultiImageHistoryDecoder,
    _interpolate_past_embeddings,
    image_history_reconstruction_loss,
    multi_image_history_reconstruction_loss,
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


def test_temporal_zarr_dataset_can_cache_normal_images_in_memory(tmp_path):
    dataset_path = tmp_path / "cached_temporal_images.zarr"
    root = zarr.open(str(dataset_path), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    images = np.arange(4 * 3 * 2 * 2, dtype=np.uint8).reshape(4, 3, 2, 2)
    data.array("action", np.zeros((4, 2), dtype=np.float32))
    data.array("agent_pose", np.zeros((4, 2), dtype=np.float32))
    data.array("full_image", images)
    meta.array("episode_ends", np.array([4], dtype=np.int64))

    dataset = TemporalZarrImageDataset(
        zarr_path=str(dataset_path),
        horizon=2,
        n_obs_steps=2,
        val_ratio=0.0,
        cache_images_in_memory=True,
    )

    assert dataset._image_cache is not None
    np.testing.assert_array_equal(dataset._sample_images(0, np.array([0, 1])), images[:2])


def test_lte_image_history_reconstruction_uses_only_causal_prefix():
    # NoT derives reconstruction time from the episode-local frame index.
    # A query at lag 0 must target the current frame and a fractional lag must
    # linearly interpolate between two earlier frames.
    history = torch.arange(5, dtype=torch.float32).reshape(1, 5, 1)
    current = torch.tensor([[3]])
    targets = _interpolate_past_embeddings(
        history, current, torch.tensor([[[0.0, 1.5]]])
    )
    torch.testing.assert_close(targets[..., 0], torch.tensor([[[3.0, 1.5]]]))

    latents = torch.randn(1, 1, 2, requires_grad=True)
    decoder = ImageHistoryDecoder(latent_dim=2, image_embedding_dim=1, hidden_dim=4)
    loss = image_history_reconstruction_loss(
        decoder, latents, history, current, num_history_queries=16
    )
    loss.backward()
    assert latents.grad is not None
    assert decoder.mlp[0].weight.grad is not None


def test_multi_image_lte_fuses_any_number_of_views_with_per_view_heads():
    fusion = MultiImageEmbeddingFusion([2, 3, 4], output_dim=5, hidden_dim=7)
    encoder = ImageNoTimeTemporalEncoder(5, latent_dim=6, hidden_dim=8)
    decoder = MultiImageHistoryDecoder(6, [2, 3, 4], hidden_dim=9)
    histories = [torch.randn(1, 4, dim) for dim in (2, 3, 4)]
    fused = fusion(histories)
    states = encoder.encode_history(fused)
    selected = states[:, [2, 3]]
    loss = multi_image_history_reconstruction_loss(
        decoder, selected, histories, torch.tensor([[2, 3]]), num_history_queries=3
    )

    assert fused.shape == (1, 4, 5)
    assert torch.isfinite(loss)
    loss.backward()
    assert fusion.mlp[0].weight.grad is not None
    assert all(head.weight.grad is not None for head in decoder.heads)
