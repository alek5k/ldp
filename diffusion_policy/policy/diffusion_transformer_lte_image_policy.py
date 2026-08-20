"""No-history diffusion-transformer decoder conditioned on LTE-IMG-NoT state."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import (
    DiffusionTransformerHybridImagePolicy,
)
from diffusion_policy.policy.lte_image_temporal_mixin import LTEImageTemporalMixin


class DiffusionTransformerLTEImagePolicy(
    LTEImageTemporalMixin, DiffusionTransformerHybridImagePolicy
):
    """Transformer decoder for LTE-IMG-NoT.

    The transformer receives the ordinary short observation window augmented
    with one recurrent LTE latent per observation. It deliberately implements
    the paper's *no-history/no-PTP* decoder objective: the long history is
    available only through LTE, and action tokens preceding the current action
    are excluded from the denoising loss.
    """

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        temporal_image_zarr_path: str | None = None,
        temporal_image_hdf5_path: str | None = None,
        temporal_rgb_key: str | None = None,
        temporal_latent_dim: int = 32,
        temporal_hidden_dim: int = 128,
        temporal_recurrent: bool = True,
        temporal_subsample_frames: int = 1,
        temporal_encode_chunk_size: int = 256,
        num_inference_steps: int | None = None,
        use_embed_if_present: bool = False,
        crop_shape=(76, 76),
        obs_encoder_group_norm: bool = True,
        eval_fixed_crop: bool = True,
        n_layer: int = 8,
        n_cond_layers: int = 0,
        n_head: int = 4,
        n_emb: int = 256,
        p_drop_emb: float = 0.0,
        p_drop_attn: float = 0.3,
        causal_attn: bool = True,
        time_as_cond: bool = True,
        obs_as_cond: bool = True,
        pred_action_steps_only: bool = False,
        image_pool_class: str = "SpatialSoftmax",
        **kwargs,
    ):
        if not obs_as_cond:
            raise ValueError("DiffusionTransformerLTEImagePolicy requires obs_as_cond=true")
        # The parent builds the Robomimic visual encoder and normalizer. Its
        # transformer is immediately replaced below with the LTE-conditioned
        # version, so no PTP controls are exposed by this policy.
        super().__init__(
            shape_meta=shape_meta,
            noise_scheduler=noise_scheduler,
            horizon=horizon,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            num_inference_steps=num_inference_steps,
            use_embed_if_present=use_embed_if_present,
            crop_shape=crop_shape,
            obs_encoder_group_norm=obs_encoder_group_norm,
            eval_fixed_crop=eval_fixed_crop,
            n_layer=n_layer,
            n_cond_layers=n_cond_layers,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=obs_as_cond,
            pred_action_steps_only=pred_action_steps_only,
            image_pool_class=image_pool_class,
            past_action_pred=False,
            past_steps_reg=-1,
            **kwargs,
        )
        self._init_lte_temporal(
            shape_meta=shape_meta,
            temporal_rgb_key=temporal_rgb_key,
            temporal_latent_dim=temporal_latent_dim,
            temporal_hidden_dim=temporal_hidden_dim,
            temporal_recurrent=temporal_recurrent,
            temporal_subsample_frames=temporal_subsample_frames,
            temporal_encode_chunk_size=temporal_encode_chunk_size,
            temporal_image_zarr_path=temporal_image_zarr_path,
            temporal_image_hdf5_path=temporal_image_hdf5_path,
        )
        self.use_embed_if_present = bool(use_embed_if_present)
        # Parent-only / legacy U-Net fields are intentionally ignored by this
        # decoder and must not be forwarded to scheduler.step.
        self.kwargs = {}
        self.model = TransformerForDiffusion(
            input_dim=self.action_dim,
            output_dim=self.action_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=self.obs_feature_dim + temporal_latent_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=True,
            n_cond_layers=n_cond_layers,
        )

    def _temporal_image_embedding_dim(self) -> int:
        encoder = self.obs_encoder.obs_nets[self.temporal_rgb_key]
        return int(encoder.output_shape(encoder.input_shape)[0])

    def _encode_temporal_images(self, images: torch.Tensor) -> torch.Tensor:
        # Apply exactly the selected camera's Robomimic preprocessing and
        # visual core; this is the transformer analogue of the LTE ResNet path.
        randomizer = self.obs_encoder.obs_randomizers[self.temporal_rgb_key]
        encoder = self.obs_encoder.obs_nets[self.temporal_rgb_key]
        return encoder(randomizer(images))

    def _condition_features(
        self, nobs: Dict[str, torch.Tensor], temporal_latents: torch.Tensor
    ) -> torch.Tensor:
        batch_size, tobs = temporal_latents.shape[:2]
        if self.use_embed_if_present and "embedding" in nobs:
            visual_features = nobs["embedding"][:, :tobs]
        else:
            obs_input = dict_apply(
                nobs, lambda value: value[:, :tobs].reshape(-1, *value.shape[2:])
            )
            visual_features = self.obs_encoder(obs_input).reshape(batch_size, tobs, -1)
        if visual_features.shape[-1] != self.obs_feature_dim:
            raise ValueError("cached embedding dimension does not match the transformer observation encoder")
        return torch.cat([visual_features, temporal_latents], dim=-1)

    def conditional_sample(self, condition_data, condition_mask, cond, generator=None, **kwargs):
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for timestep in self.noise_scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = self.model(trajectory, timestep, cond)
            trajectory = self.noise_scheduler.step(
                model_output, timestep, trajectory, generator=generator, **kwargs
            ).prev_sample
        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        batch_size = nobs[self.temporal_rgb_key].shape[0]
        temporal_latents = self._online_temporal_latents(nobs)
        cond = self._condition_features(nobs, temporal_latents)
        condition_data = torch.zeros(
            (batch_size, self.horizon, self.action_dim), device=self.device, dtype=self.dtype
        )
        condition_mask = torch.zeros_like(condition_data, dtype=torch.bool)
        sample = self.conditional_sample(condition_data, condition_mask, cond, **self.kwargs)
        action_pred = self.normalizer["action"].unnormalize(sample)
        start = self.n_obs_steps - 1
        return {
            "action": action_pred[:, start:start + self.n_action_steps],
            "action_pred": action_pred,
        }

    def get_optimizer(
        self,
        transformer_weight_decay: float,
        obs_encoder_weight_decay: float,
        learning_rate: float | None = None,
        betas=(0.9, 0.95),
        lr: float | None = None,
        **_ignored,
    ):
        """Use the released transformer optimizer schema.

        The LTE configs inherit shared U-Net defaults, whose ``_target_``,
        ``lr`` and ``eps`` keys are intentionally ignored here.
        """
        if learning_rate is None:
            if lr is None:
                raise ValueError("learning_rate is required for the transformer optimizer")
            learning_rate = lr
        return super().get_optimizer(
            transformer_weight_decay=transformer_weight_decay,
            obs_encoder_weight_decay=obs_encoder_weight_decay,
            learning_rate=learning_rate,
            betas=tuple(betas),
        )

    def compute_loss(self, batch, debug: bool = False):
        nobs = self.normalizer.normalize(batch["obs"])
        actions = self.normalizer["action"].normalize(batch["action"])
        batch_size = actions.shape[0]
        temporal_latents = self._history_latents_from_batch(batch)
        cond = self._condition_features(nobs, temporal_latents)
        condition_mask = self.mask_generator(actions.shape)
        noise = torch.randn_like(actions)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=actions.device,
        ).long()
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
        noisy_actions[condition_mask] = actions[condition_mask]
        pred = self.model(noisy_actions, timesteps, cond)
        target = noise if self.noise_scheduler.config.prediction_type == "epsilon" else actions
        if self.noise_scheduler.config.prediction_type not in {"epsilon", "sample"}:
            raise ValueError("Unsupported diffusion prediction type")
        # No PTP: do not train reconstructed action positions preceding the
        # current action. LTE is the only long-history mechanism.
        start = self.n_obs_steps - 1
        pred, target, condition_mask = pred[:, start:], target[:, start:], condition_mask[:, start:]
        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * (~condition_mask).type(loss.dtype)
        return reduce(loss, "b ... -> b (...)", "mean").mean()
