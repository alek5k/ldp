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

## Current TC-DP paper comparison: three training seeds

Use training seeds 42, 43, and 44 for both WaitAtGoal and LiftQA. The saved
configurations are in `experiment_configs/paper_ptp_seeds/20261029_234821/`.
Each environment has seed-43 and seed-44 configurations; the example below
uses the seed-43 configuration as a template and overrides the training seed
for all three runs. The dataset split seed stays fixed at 42.

The paper comparison uses 16 adjacent observations, a 32-token prediction
horizon, eight executed actions, full PTP (`past_action_pred=true`,
`past_steps_reg=-1`), batch size 128, and 500 epochs. It uses full 144x144 images
with no crop and `SpatialMeanPool`. Online training rollouts are disabled;
checkpoint selection uses held-out validation loss.

From the repository root, set the local demonstration path and run:

```bash
ENV=liftqa
DATASET=/path/to/lift_qa_v2.zarr
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy NUMBA_DISABLE_JIT=1
for SEED in 42 43 44; do
    RUN="${ENV}_ptp_seed${SEED}"
    python train.py \
        --config-dir=experiment_configs/paper_ptp_seeds/20261029_234821 \
        --config-name="${ENV}_seed43" \
        training.seed="$SEED" training.device=cuda:0 \
        task.dataset.zarr_path="$DATASET" \
        hydra.run.dir="data/outputs/$RUN" multi_run.run_dir="data/outputs/$RUN" \
        logging.name="$RUN" logging.mode=disabled
done
```

Repeat with `ENV=waitatgoal` and the local path to
`wait_at_goal_dataset_v4.zarr`. Select the checkpoint with the lowest validation
loss separately for each training seed, then evaluate each selected checkpoint
on the same 200 evaluation seeds:

```bash
ENV=liftqa
SEED=42
CHECKPOINT=/path/to/selected_validation_checkpoint.ckpt
./eval_temporal_policy.sh "$ENV" "$CHECKPOINT" "data/inference/${ENV}_ptp_seed${SEED}" \
    --n_test=200 --n_test_vis=0 --test_start_seed=1000 \
    --max_steps=400 --num_inference_steps=100
```

Repeat evaluation for training seeds 43 and 44, and for WaitAtGoal. Evaluation
is a separate step; training does not launch it automatically.

The current comparison table is generated in TemporalDiffusionPolicy using
`python -m temporal_diffusion_policy.analysis.report` (the default `paper_main`
preset), or `--config main_multiseed` for the simulation-only comparison. Point
its `tc_dp_paper_multiseed` study lists at the three `rollouts.zarr` outputs per
environment. The report computes each metric per training seed and reports
the mean and sample standard deviation. The lower learning-rate LiftQA trial
is a separate exploratory experiment and is not part of this comparison.

## Other single-run presets

The training launcher always enables full PTP (`past_action_pred=true`,
`past_steps_reg=-1`), validates every 10 epochs, and ranks checkpoints by
held-out `val_loss`. It enables online rollouts every 50 epochs by default;
the commands below also launch a separate Zarr evaluation after training.
These older examples use evaluation seeds starting at 200; use the
three-seed protocol above for the current paper comparison.

### Short-history comparison

This exploratory preset matches the short-history TC-DP inference horizons: two
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
adjacent rather than temporally spaced. Batch size 128 keeps the 16-image visual
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
