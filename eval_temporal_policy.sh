#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 {waitatgoal|liftqa} CHECKPOINT [OUTPUT_DIR] [eval.py options...]" >&2
    exit 2
fi

env_name="$1"
checkpoint="$2"
output_dir="${3:-data/inference/${env_name}_$(basename "${checkpoint%.ckpt}")_eval}"

case "$env_name" in
    waitatgoal|liftqa) ;;
    *) echo "Unknown environment: $env_name" >&2; exit 2 ;;
esac

# Safe defaults for SSH / headless evaluation. Set either variable before
# invoking this script to use a different rendering backend.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"

# Videos are off by default because checkpoint selection needs scores and Zarr
# episodes, not MP4 encoding. Trailing eval.py options can override these.
python eval.py \
    --checkpoint="$checkpoint" \
    --output_dir="$output_dir" \
    --zarr_path="$output_dir/rollouts.zarr" \
    --n_test=200 --n_test_vis=0 --test_start_seed=200 \
    --max_steps=400 --n_action_steps=8 --num_inference_steps=100 \
    "${@:4}"
