# Square configuration variants

The main Square diffusion-transformer configuration is
`transformer_square.yaml`. Its temporal settings are:

| Setting | Value |
| --- | ---: |
| `global_horizon` | 32 |
| `global_obs` | 2 |
| `global_action` | 1 |
| `subsample_frames` | 1 |

The transformer variants keep these settings unless noted otherwise.

| Config | Difference from the standard transformer config |
| --- | --- |
| `transformer_square.yaml` | Standard diffusion-transformer PTP setup using `image_abs.hdf5`. |
| `transformer_square_emb.yaml` | Uses the frozen `obs_encoders/square_encoder.ckpt`; `pad_before=128`; disables the dataset cache. |
| `transformer_square_past.yaml` | Uses `image_abs_past.hdf5`, adding a 7-D `past_act` field; disables the dataset cache. |
| `transformer_square_past_emb.yaml` | Uses `image_abs_past_emb.hdf5`; its embedding is 144-D rather than 137-D because it includes the 7-D past action; disables the dataset cache. |
| `transformer_square_reg.yaml` | Replaces the diffusion policy with `RegressionTransformerHybridImagePolicy`, so it has no diffusion noise scheduler. |
| `transformer_square_reg_emb.yaml` | Regression transformer with frozen `obs_encoders/square_l1_encoder.ckpt`; no diffusion scheduler, `pad_before=64`, and dataset cache disabled. |
| `transformer_square_bc_rnn.yaml` | Separate Robomimic image RNN/behaviour-cloning baseline, not a transformer. Its effective `horizon` and `dataset_obs_steps` are both 10. |

All transformer variants use two observed frames and execute one action at a
time. The main experimental axes are diffusion versus regression, raw images
versus frozen image embeddings, and the optional 7-D past-action feature.
