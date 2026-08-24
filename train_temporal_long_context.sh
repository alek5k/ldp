#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 {waitatgoal|liftqa} [OUTPUT_DIR] [Hydra overrides...]" >&2
    exit 2
fi

env_name="$1"
case "$env_name" in
    waitatgoal)
        task_name=waitatgoal_image
        dataset_path=/mnt/shared/TemporalDiffusionPolicy/data/demonstration/wait_at_goal_dataset_v4.zarr
        ;;
    liftqa)
        task_name=liftqa_image
        dataset_path=/mnt/shared/TemporalDiffusionPolicy/data/demonstration/lift_qa_v2.zarr
        ;;
    *) echo "Unknown environment: $env_name" >&2; exit 2 ;;
esac
output_dir="${2:-data/outputs/${env_name}_ptp_$(date +%Y%m%d_%H%M%S)}"

# LiftQA uses MuJoCo off-screen rendering. EGL is the reliable headless backend
# on the training machine; users can still override it before invoking this script.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
# WaitAtGoal renders observations with Pygame; use its headless backend over SSH.
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
# Avoid a several-minute Numba compilation before the first sparse Zarr batch.
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"

# Temporal-comparison visual defaults: use full frames and mean pooling. To
# restore LDP's original visual preprocessing, invoke this script with
# LDP_CROP_SHAPE='[76,76]' LDP_IMAGE_POOL_CLASS=SpatialSoftmax.
temporal_crop_shape="${LDP_CROP_SHAPE:-null}"
temporal_image_pool_class="${LDP_IMAGE_POOL_CLASS:-SpatialMeanPool}"

# Evaluate online every 50 epochs while retaining validation-loss checkpoint
# selection for the temporal-comparison runs.
# Match TemporalDiffusionPolicy inference: predict 16 tokens from two
# observations and execute the first eight actions before replanning.
python train.py --config-name=train_diffusion_transformer_hybrid_workspace \
    task="$task_name" \
    task.dataset.zarr_path="$dataset_path" \
    horizon=16 n_obs_steps=2 n_action_steps=8 \
    +policy.past_action_pred=true +policy.past_steps_reg=-1 \
    policy.crop_shape="$temporal_crop_shape" \
    +policy.image_pool_class="$temporal_image_pool_class" \
    dataloader.batch_size=512 dataloader.num_workers=2 dataloader.pin_memory=false \
    val_dataloader.batch_size=512 val_dataloader.num_workers=2 val_dataloader.pin_memory=false \
    training.gradient_accumulate_every=1 \
    task.dataset.val_ratio=0.2 \
    training.num_epochs=200 training.lr_warmup_steps=500 \
    training.val_every=10 training.checkpoint_every=50 training.rollout_every=50 \
    checkpoint.topk.monitor_key=val_loss checkpoint.topk.mode=min \
    'checkpoint.topk.format_str="epoch={epoch:04d}-val_loss={val_loss:.3f}.ckpt"' \
    hydra.run.dir="$output_dir" \
    "${@:3}"
