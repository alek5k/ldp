# Paper-reproduction PTP configs

The `*_paper.yaml` files reproduce the long-context diffusion + PTP settings
reported in *Learning Long-Context Diffusion Policies via Past-Token
Prediction*. They compose the corresponding `*_emb.yaml` configuration, which
freezes a short-context visual encoder and consumes cached image embeddings.
For the raw Robomimic tasks (Square, Tool Hang, and Transport), the experiment
CLI and VM training script regenerate the required `embedding` field with
`rewrite_with_embeddings.py` before launching training. The supplied
long-horizon ALOHA and Square datasets are used as-is; they are not rewritten.

| Task | Observations | Horizon | Action chunk | Subsampling | Epochs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Square | 16 | 32 | 8 | 1 | 500 |
| Tool Hang | 16 | 32 | 8 | 1 | 500 |
| Transport | 16 | 32 | 8 | 1 | 500 |
| Long-Horizon Square | 20 | 32 | 8 | 20 | 500 |
| Long-Horizon ALOHA | 32 | 48 | 8 | 20 | 1500 |

The paper's primary simulation comparison uses action chunks of eight. For a
fully closed-loop experiment, override `global_action=1` rather than changing
the reproduction config.

The paper explicitly gives horizon 32 for 16 observations. Long-Horizon Square
keeps the public configuration's horizon 32, leaving 12 future tokens for its
20 observations. LH ALOHA uses horizon 48 to avoid its otherwise zero-length
future window at 32 observations.
