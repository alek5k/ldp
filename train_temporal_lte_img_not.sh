#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 {waitatgoal|liftqa} [OUTPUT_DIR] [Hydra overrides...]" >&2
    exit 2
fi

case "$1" in
    waitatgoal|liftqa) task_name="${1}_image" ;;
    *) echo "Unknown environment: $1" >&2; exit 2 ;;
esac
output_dir="${2:-data/outputs/${1}_lte_img_not}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"

python train.py --config-name=train_diffusion_unet_lte_image_workspace \
    task="$task_name" \
    hydra.run.dir="$output_dir" \
    "${@:3}"
