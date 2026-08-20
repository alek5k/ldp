# Temporal environments in LDP

`diffusion_policy.env.waitatgoal.WaitAtGoal` and
`diffusion_policy.env.liftqa.LiftQA` are native copies of the environments
previously maintained in TemporalDiffusionPolicy. Training reads the existing
TemporalDiffusionPolicy Zarr episodes directly; no data collection or HDF5
conversion is required.

## Environment

```bash
conda activate robodiff-lh-5090
cd /home/sydney1/Repos/ldp
```

The launchers default to `MUJOCO_GL=egl`, `SDL_VIDEODRIVER=dummy`, and
`NUMBA_DISABLE_JIT=1`, so they work over SSH. They also default to full 144×144
frames, no crop, and `SpatialMeanPool` rather than SpatialSoftmax. The commands
below set the visual options explicitly so an exported shell variable cannot
change the experiment accidentally.

## Run presets

The training launcher always enables full PTP (`past_action_pred=true`,
`past_steps_reg=-1`), validates every 10 epochs, and ranks checkpoints by
held-out `val_loss`. Training rollouts are disabled; each command runs the
fixed-seed Zarr evaluation immediately afterward.

### Fair TC-DP comparison

Use this for a direct comparison with the current TC-DP inference setup: two
adjacent observations, 16 predicted tokens, eight executed actions, batch size
512, and 200 epochs. It has one PTP past-action target and 14 future-action
targets.

```bash
GPU=0 ENV=liftqa HORIZON=16 N_OBS=2 SUBSAMPLE=1 N_ACTION=8 EPOCHS=200 BATCH=512; RUN=${ENV}_h${HORIZON}_o${N_OBS}_s${SUBSAMPLE}_a${N_ACTION}_e${EPOCHS}; LDP_CROP_SHAPE=null LDP_IMAGE_POOL_CLASS=SpatialMeanPool CUDA_VISIBLE_DEVICES=$GPU ./train_temporal_long_context.sh $ENV data/outputs/$RUN horizon=$HORIZON n_obs_steps=$N_OBS n_action_steps=$N_ACTION task.dataset.subsample_frames=$SUBSAMPLE task.env_runner.subsample_frames=$SUBSAMPLE dataloader.batch_size=$BATCH val_dataloader.batch_size=$BATCH training.num_epochs=$EPOCHS && CUDA_VISIBLE_DEVICES=$GPU ./eval_temporal_policy.sh $ENV data/outputs/$RUN/checkpoints/latest.ckpt data/inference/${RUN}_eval --n_test=200 --n_test_vis=10 --test_start_seed=200 --max_steps=400 --num_inference_steps=100
```

Set `ENV=waitatgoal` to run the same fair condition on WaitAtGoal.

### PTP-focused long-history runs

These are deliberately not strict TC-DP-parity runs. They use the paper's
long-context table row: 16 adjacent observations, a 32-token action horizon,
and 16 future tokens. Full PTP therefore supervises 15 past actions, the
current action, and 16 future actions. `n_action_steps=8` is separate: it only
controls how many predicted actions are executed before replanning.

Both environments use `subsample_frames=1`, so all 16 observations are
adjacent rather than temporally spaced. Batch size 64 keeps the 16-image visual
context practical; 500 epochs follows the paper's simulation-training
duration.

#### WaitAtGoal

```bash
cd ~/Repos/ldp && conda activate robodiff-lh-5090 
GPU=1 ENV=waitatgoal HORIZON=32 N_OBS=16 SUBSAMPLE=1 N_ACTION=8 EPOCHS=500 BATCH=128; RUN=${ENV}_h${HORIZON}_o${N_OBS}_s${SUBSAMPLE}_a${N_ACTION}_e${EPOCHS}; LDP_CROP_SHAPE=null LDP_IMAGE_POOL_CLASS=SpatialMeanPool CUDA_VISIBLE_DEVICES=$GPU ./train_temporal_long_context.sh $ENV data/outputs/$RUN horizon=$HORIZON n_obs_steps=$N_OBS n_action_steps=$N_ACTION task.dataset.subsample_frames=$SUBSAMPLE task.env_runner.subsample_frames=$SUBSAMPLE dataloader.batch_size=$BATCH val_dataloader.batch_size=$BATCH training.num_epochs=$EPOCHS && CUDA_VISIBLE_DEVICES=$GPU ./eval_temporal_policy.sh $ENV data/outputs/$RUN/checkpoints/latest.ckpt data/inference/${RUN}_eval --n_test=200 --n_test_vis=10 --test_start_seed=200 --max_steps=400 --num_inference_steps=100
```

#### LiftQA

```bash
cd ~/Repos/ldp && conda activate robodiff-lh-5090 
GPU=0 ENV=liftqa HORIZON=32 N_OBS=16 SUBSAMPLE=1 N_ACTION=8 EPOCHS=500 BATCH=128; RUN=${ENV}_h${HORIZON}_o${N_OBS}_s${SUBSAMPLE}_a${N_ACTION}_e${EPOCHS}; LDP_CROP_SHAPE=null LDP_IMAGE_POOL_CLASS=SpatialMeanPool CUDA_VISIBLE_DEVICES=$GPU ./train_temporal_long_context.sh $ENV data/outputs/$RUN horizon=$HORIZON n_obs_steps=$N_OBS n_action_steps=$N_ACTION task.dataset.subsample_frames=$SUBSAMPLE task.env_runner.subsample_frames=$SUBSAMPLE dataloader.batch_size=$BATCH val_dataloader.batch_size=$BATCH training.num_epochs=$EPOCHS && CUDA_VISIBLE_DEVICES=$GPU ./eval_temporal_policy.sh $ENV data/outputs/$RUN/checkpoints/latest.ckpt data/inference/${RUN}_eval --n_test=200 --n_test_vis=10 --test_start_seed=200 --max_steps=400 --num_inference_steps=100
```

cd ~/Repos/ldp && conda activate robodiff-lh-5090 
GPU=1 ENV=liftqa HORIZON=32 N_OBS=16 SUBSAMPLE=1 N_ACTION=8 EPOCHS=500 BATCH=128; RUN=liftqa_h32_o16_s1_a8_e500; LDP_CROP_SHAPE=null LDP_IMAGE_POOL_CLASS=SpatialMeanPool && CUDA_VISIBLE_DEVICES=$GPU ./eval_temporal_policy.sh $ENV data/outputs/$RUN/checkpoints/epoch=0249-val_loss=0.142.ckpt data/inference/${RUN}_epoch250_eval --n_test=200 --n_test_vis=10 --test_start_seed=200 --max_steps=400 --num_inference_steps=100


### Why embeddings are not cached

The paper caches frozen visual-encoder features only to avoid recomputing them
during training. With identical preprocessing and optimization, that does not
improve task success. These runs therefore train end-to-end from the original
Zarr images; evaluation always encodes newly generated environment images
online in any case.

## Evaluation output

`eval_temporal_policy.sh` is the supported evaluation entry point. It applies
SSH-safe rendering defaults and writes `$OUTPUT_DIR/rollouts.zarr` in the same
`data/` and `meta/episode_ends` layout as TemporalDiffusionPolicy. Each entry
has the pre-action environment observation (`full_image` and `agent_pose`),
action, reward, and done flag. WaitAtGoal also records `wait_time`,
`step_count`, and `wait_times_each_visit`. The presets evaluate 200 fixed-seed
test episodes and record MP4s for the first 10.

The source datasets are:

- `/mnt/shared/TemporalDiffusionPolicy/data/demonstration/wait_at_goal_dataset_v4.zarr`
- `/mnt/shared/TemporalDiffusionPolicy/data/demonstration/lift_qa_v2.zarr`

## LTE-IMG-NoT

`train_temporal_lte_img_not.sh` trains the migrated learned temporal encoder
against these native environments.  It preserves the original method's
causal state update: a detached ResNet image feature and the previous latent
produce the next latent, with no timestep input.  Training obtains the full
episode prefix from lightweight Zarr frame indices; rollout updates the state
after every executed action, including actions between replans.

```bash
cd ~/Repos/ldp && conda activate robodiff-lh-5090
GPU=0 CUDA_VISIBLE_DEVICES=$GPU ./train_temporal_lte_img_not.sh waitatgoal data/outputs/waitatgoal_lte_img_not
CUDA_VISIBLE_DEVICES=$GPU ./eval_temporal_policy.sh waitatgoal data/outputs/waitatgoal_lte_img_not/checkpoints/latest.ckpt data/inference/waitatgoal_lte_img_not_eval
```

The default batch size is 16 because every batch encodes causal image prefixes
online.  Change `dataloader.batch_size` only after checking GPU memory.

### LDP image benchmarks

LTE-IMG-NoT also runs directly on LDP's existing Square, ToolHang, Transport,
long-history ALOHA, and long-history Square environments. The standalone
configs under `experiment_configs/` contain the benchmark dataset and runner
settings rather than modifying generic task defaults. The normal policy encoder
retains every task camera; LTE uses one declared primary camera to match the
original single-ResNet-image temporal input:

| Launcher task | Default experiment config | Dataset | LTE camera |
| --- | --- | --- | --- |
| `square` | `square/lte_img_not_transformer.yaml` | Square MH, absolute actions | `agentview_image` |
| `tool_hang` | `tool/lte_img_not_transformer.yaml` | Tool Hang PH, absolute actions | `sideview_image` |
| `transport` | `transport/lte_img_not_transformer.yaml` | Transport MH, absolute actions | `shouldercamera0_image` |
| `lh-aloha` | `aloha/lte_img_not_transformer.yaml` | long-horizon ALOHA | `top` |
| `lh-square` | `longhist/lte_img_not_transformer.yaml` | long-horizon Square | `agentview_image` |

```bash
cd ~/Repos/ldp && conda activate robodiff-lh-5090
CUDA_VISIBLE_DEVICES=0 ./train_lte_img_not.sh square data/outputs/square_lte_img_not
CUDA_VISIBLE_DEVICES=0 ./train_lte_img_not.sh tool_hang data/outputs/tool_hang_lte_img_not
CUDA_VISIBLE_DEVICES=0 ./train_lte_img_not.sh transport data/outputs/transport_lte_img_not
CUDA_VISIBLE_DEVICES=0 ./train_lte_img_not.sh lh-aloha data/outputs/lh_aloha_lte_img_not
CUDA_VISIBLE_DEVICES=0 ./train_lte_img_not.sh lh-square data/outputs/lh_square_lte_img_not
```

The transformer decoder is the default. It receives two adjacent observations;
LTE alone carries longer history, and no PTP past-action objective is used. Add
the `-unet` suffix to any launcher task to select the original U-Net decoder.

For interactive launch and evaluation in a detached `screen` session, run
`python experiment_cli.py`. It offers `train`, `eval`, and `train+eval`; each
mode asks for the number of sequential runs and assigns consecutive seeds
starting at 42 by default.

```bash
MUJOCO_GL=egl MPLCONFIGDIR=/tmp/ldp_mpl CUDA_VISIBLE_DEVICES=0 \
  ./train_lte_img_not.sh lh-aloha data/outputs/lh_aloha_lte_transformer_seed42 \
  training.seed=42 training.device=cuda:0
MUJOCO_GL=egl MPLCONFIGDIR=/tmp/ldp_mpl CUDA_VISIBLE_DEVICES=0 \
  ./train_lte_img_not.sh lh-square data/outputs/lh_square_lte_transformer_seed42 \
  training.seed=42 training.device=cuda:0
```

`train_lte_img_not.sh` takes `TASK OUTPUT_DIR [Hydra overrides...]`. Use a
different output directory for every seed, so checkpoints and logs do not
overwrite each other. For example, a second LH-Square seed is:

```bash
MUJOCO_GL=egl MPLCONFIGDIR=/tmp/ldp_mpl CUDA_VISIBLE_DEVICES=0 \
  ./train_lte_img_not.sh lh-square data/outputs/lh_square_lte_transformer_seed43 \
  training.seed=43 training.device=cuda:0
```

These presets use the task config's action-replanning horizon. They use the
corresponding released experiment's rollout cadence and retain checkpoints by
maximum `test_mean_score`, rather than by offline validation loss. LTE keeps a
batch size of 16 in the U-Net presets because it encodes causal image prefixes
online; the aligned transformer presets use batch size 64.
