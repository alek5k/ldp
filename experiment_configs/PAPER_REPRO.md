# Paper-reproduction PTP configs

The `*_paper.yaml` files reproduce the long-context diffusion + PTP settings
reported in *Learning Long-Context Diffusion Policies via Past-Token
Prediction*. They compose the corresponding `*_emb.yaml` configuration, which
freezes a short-context visual encoder. Run the short-context encoder-pretrain
stage and generate its cached image embeddings before using these configs.

| Task | Observations | Horizon | Action chunk | Subsampling | Epochs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Square | 16 | 32 | 8 | 1 | 500 |
| Tool Hang | 16 | 32 | 8 | 1 | 500 |
| Transport | 16 | 32 | 8 | 1 | 500 |
| Long-Horizon Square | 20 | 32 | 8 | 20 | 500 |
| Long-Horizon ALOHA | 32 | 32 | 8 | 20 | 1500 |

The paper's primary simulation comparison uses action chunks of eight. For a
fully closed-loop experiment, override `global_action=1` rather than changing
the reproduction config.

The appendix explicitly reports the observation count, subsampling rate, and
epoch count for the two long-horizon tasks, but not a separate prediction
horizon for their 20- and 32-observation settings. Those configs retain the
public implementation's `global_horizon=32`.
