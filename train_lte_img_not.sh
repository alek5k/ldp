#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 {square|tool_hang|transport|lh-aloha|lh-square}[[-unet]] [OUTPUT_DIR] [Hydra overrides...]" >&2
    exit 2
fi

case "$1" in
    square) config_name=square/lte_img_not_transformer ;;
    tool_hang) config_name=tool/lte_img_not_transformer ;;
    transport) config_name=transport/lte_img_not_transformer ;;
    lh-aloha) config_name=aloha/lte_img_not_transformer ;;
    lh-square) config_name=longhist/lte_img_not_transformer ;;
    square-unet) config_name=square/lte_img_not ;;
    tool_hang-unet) config_name=tool/lte_img_not ;;
    transport-unet) config_name=transport/lte_img_not ;;
    lh-aloha-unet) config_name=aloha/lte_img_not ;;
    lh-square-unet) config_name=longhist/lte_img_not ;;
    *) echo "Unknown LTE task: $1" >&2; exit 2 ;;
esac
output_dir="${2:-data/outputs/${1}_lte_img_not}"

python train.py --config-dir=experiment_configs --config-name="$config_name" \
    hydra.run.dir="$output_dir" \
    "${@:3}"
