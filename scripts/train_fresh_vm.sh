#!/usr/bin/env bash
# Train an LDP run after scripts/setup_fresh_vm.sh has prepared the VM.
set -Eeuo pipefail

###############################################################################
# QUICK CONFIGURATION -- edit this section for each training job.
###############################################################################

TARGET_ENVIRONMENT="square"           # square | transport | tool-hang | lh-aloha | lh-square
TRAINING_METHOD="ptp"                 # lte | ptp
LTE_ARCHITECTURE="unet"               # transformer | unet
TRAIN_BATCH_SIZE=64
TRAIN_EPOCHS="auto"                  # auto: retain the selected config's epoch count.
RUN_NOTE=""                          # Written to note.txt for each launched run; blank disables it.

LTE_IMAGE_COUNT=1                     # First N RGB cameras for the target environment.
LTE_HISTORY_DECODER_SAMPLES=16
LTE_TEMPORAL_LATENT_DIM=64
LTE_TEMPORAL_HIDDEN_DIM=256
LTE_TEMPORAL_HIDDEN_LAYERS=1
LTE_HISTORY_DECODER_HIDDEN_DIM=256
LTE_HISTORY_DECODER_HIDDEN_LAYERS=1
LTE_EMBEDDING_CACHE=true
LTE_CACHE_START_EPOCH=5
LTE_CACHE_WARMUP_EPOCHS=20
LTE_CACHE_REFRESH_EPOCHS=5

DOWNLOAD_OBS_ENCODERS=true            # Required by the default PTP paper configs.

TRAIN_GPU="0"

TRAIN_DATALOADER_WORKERS=8
TRAIN_PIN_MEMORY=true
TRAIN_PERSISTENT_WORKERS=true
TRAIN_PREFETCH_FACTOR=1
TRAIN_IMAGE_AUGMENTATION=false          # ColorJitter; can slow CPU-bound training.
TRAIN_CACHE_IMAGES_ON_GPU=true          # Faster input pipeline; consumes GPU RAM.
TRAIN_LEARNING_RATE="0.0001"
TRAIN_SEED=42
TRAIN_SEQUENTIAL_RUNS=1
SHUTDOWN_AFTER_COMPLETION=false

###############################################################################
# MACHINE LOCATIONS -- normally leave unchanged.
###############################################################################

REPO_DIR="$HOME/ldp"
CONDA_DIR="$HOME/miniforge3"
CONDA_ENV_NAME="robodiff-lh-5090"
# Empty chooses the candidate filesystem with the most available space.
# setup_fresh_vm.sh links ~/ldp/data there for configs using relative paths.
DATA_ROOT=""
EXTERNAL_OUTPUT_ROOT=""
AUTH_FILE="$REPO_DIR/.vm_auth.env"

###############################################################################

log() { printf '\n==> %s\n' "$*"; }
warn() { printf '\nWARNING: %s\n' "$*" >&2; }
as_root() { if [[ "${EUID}" -eq 0 ]]; then "$@"; else sudo "$@"; fi; }

select_data_root() {
    local filesystem type blocks used available capacity mount best_available=-1 best_root=""
    while read -r filesystem type blocks used available capacity mount; do
        case "${type}" in tmpfs|devtmpfs|squashfs|proc|sysfs|cgroup*|devpts) continue ;; esac
        [[ "${available}" =~ ^[0-9]+$ ]] || continue
        if (( available > best_available )); then
            best_available="${available}"
            best_root="${mount}"
        fi
    done < <(df -PTk)
    [[ -n "${best_root}" ]] || return 1
    if [[ "${best_root}" == "/" ]]; then
        printf '%s/data\n' "${REPO_DIR}"
    else
        printf '%s/ldp/data\n' "${best_root}"
    fi
}

DATA_ROOT="${DATA_ROOT:-$(select_data_root)}"

if [[ ! -r "${AUTH_FILE}" ]]; then
    echo "Missing ${AUTH_FILE}; run scripts/setup_fresh_vm.sh first." >&2
    exit 1
fi
# Created by setup_fresh_vm.sh with mode 600; contains W&B and ntfy credentials.
source "${AUTH_FILE}"

on_exit() {
    local exit_status=$?
    trap - EXIT
    set +e
    if (( exit_status != 0 )); then
        [[ -n "${NTFY_AUTH_TOKEN:-}" ]] && curl --fail --silent --show-error --max-time 20 \
            -H "Authorization: Bearer ${NTFY_AUTH_TOKEN}" -H "Title: LDP training failed" -H "Tags: warning" \
            --data-binary "LDP training failed on $(hostname): exit_status=${exit_status}; env=${TARGET_ENVIRONMENT}; method=${TRAINING_METHOD}" \
            "${NTFY_SERVER%/}/${NTFY_TOPIC}" || true
        [[ "${SHUTDOWN_AFTER_COMPLETION}" == "true" ]] && as_root shutdown -h now || true
    fi
    exit "${exit_status}"
}
trap on_exit EXIT

source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
cd "${REPO_DIR}"
log "Dataset root selected by free space: ${DATA_ROOT}"
# The setup script installs the persistent memlock limit. On managed VMs this
# shell may not be allowed to raise it itself, which is harmless for training.
ulimit -l unlimited 2>/dev/null || true

log "Logging into Weights & Biases"
export WANDB_API_KEY WANDB_ENTITY
python -c 'import os, wandb; wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)'

download_google_drive_zip() {
    local file_id="$1" archive_name="$2" expected_path="$3"
    local extract_root="${4:-${DATA_ROOT}}" force_download="${5:-false}" archive_path cookie_file html_file uuid confirm final_url
    archive_path="${extract_root}/${archive_name}"
    if [[ "${force_download}" != "true" && -e "${expected_path}" ]]; then
        log "Dataset already present: ${expected_path}"
        return
    fi
    log "Downloading ${archive_name} from Google Drive"
    mkdir -p "${extract_root}"
    cookie_file="$(mktemp)"; html_file="$(mktemp)"
    wget --quiet --save-cookies "${cookie_file}" --keep-session-cookies \
        "https://drive.google.com/uc?export=download&id=${file_id}" -O "${html_file}"
    uuid="$(grep -oP 'name="uuid" value="\K[^"]+' "${html_file}" || true)"
    confirm="$(grep -oP 'name="confirm" value="\K[^"]+' "${html_file}" || true)"
    if [[ -z "${uuid}" || -z "${confirm}" ]]; then
        rm -f "${cookie_file}" "${html_file}"
        echo "Google Drive confirmation changed or requires a browser. Download manually from README.md." >&2
        return 1
    fi
    final_url="https://drive.usercontent.google.com/download?id=${file_id}&export=download&confirm=${confirm}&uuid=${uuid}"
    wget --show-progress --load-cookies "${cookie_file}" "${final_url}" -O "${archive_path}"
    rm -f "${cookie_file}" "${html_file}"
    unzip -q -o "${archive_path}" -d "${extract_root}"
    rm -f "${archive_path}"
}

download_robomimic_image() {
    local expected_path="$1" archive_path="${DATA_ROOT}/robomimic_image.zip"
    if [[ -e "${expected_path}" ]]; then
        log "Dataset already present: ${expected_path}"
        return
    fi
    log "Downloading the Robomimic image archive"
    wget --show-progress --continue \
        "https://diffusion-policy.cs.columbia.edu/data/training/robomimic_image.zip" \
        -O "${archive_path}"
    unzip -q -o "${archive_path}" -d "${DATA_ROOT}"
    rm -f "${archive_path}"
}

download_selected_dataset() {
    case "${TARGET_ENVIRONMENT}" in
        square) download_robomimic_image "${DATA_ROOT}/robomimic/datasets/square/mh/image_abs.hdf5" ;;
        tool-hang|tool_hang) download_robomimic_image "${DATA_ROOT}/robomimic/datasets/tool_hang/ph/image_abs.hdf5" ;;
        transport) download_robomimic_image "${DATA_ROOT}/robomimic/datasets/transport/mh/image_abs.hdf5" ;;
        lh-aloha) download_google_drive_zip "1gwzIRBmn0a4Orj2okMNQ9qiPPpxmqdKA" "aloha_twomodes_single.zip" "${DATA_ROOT}/aloha_twomodes_single/demos.hdf5" ;;
        lh-square) download_google_drive_zip "1-ZDi8-aVx1I8aZCan-vXJQIpLyCCNwym" "longhistsquare100.zip" "${DATA_ROOT}/longhistsquare100/demos.hdf5" ;;
        *) warn "No public dataset download is configured for ${TARGET_ENVIRONMENT}; expecting it under ${DATA_ROOT}." ;;
    esac
}

download_observation_encoders() {
    [[ "${TRAINING_METHOD}" == "ptp" && "${DOWNLOAD_OBS_ENCODERS}" == "true" ]] || return 0
    # README.md's shared archive is extracted before every PTP run.
    download_google_drive_zip "1tSYyWg3HZbTtEhzpAXQpl28DSrWsXc7J" "obs_encoders.zip" "${REPO_DIR}/obs_encoders" "${REPO_DIR}" true
}

download_selected_dataset
download_observation_encoders

next_available_run_name() {
    local output_root="$1" base_name="$2" candidate="$2" attempt=2
    while [[ -e "${output_root}/${candidate}" || -L "${output_root}/${candidate}" || \
        -e "${DATA_ROOT}/outputs/${candidate}" || -L "${DATA_ROOT}/outputs/${candidate}" || \
        -e "${DATA_ROOT}/inference/${candidate}" || -L "${DATA_ROOT}/inference/${candidate}" ]]; do
        candidate="${base_name}-r${attempt}"
        ((attempt++))
    done
    printf '%s\n' "${candidate}"
}

run_training() {
    local output_root run_name output_dir train_task lte_keys config_dir config_name timestamp
    local -a overrides available
    output_root="${EXTERNAL_OUTPUT_ROOT:-${DATA_ROOT}/outputs}"
    mkdir -p "${output_root}"
    timestamp="$(date +%Y%m%d_%H%M%S)"
    train_task="${TARGET_ENVIRONMENT/tool-hang/tool_hang}"
    if [[ "${TRAINING_METHOD}" == lte ]]; then
        if [[ "${LTE_ARCHITECTURE}" != transformer ]]; then
            case "${train_task}" in square|tool_hang|transport|lh-aloha|lh-square) train_task+="-unet" ;; esac
        fi
        case "${TARGET_ENVIRONMENT}" in
            square|lh-square) available=(agentview_image robot0_eye_in_hand_image) ;;
            tool-hang) available=(sideview_image robot0_eye_in_hand_image) ;;
            transport) available=(shouldercamera0_image shouldercamera1_image robot0_eye_in_hand_image) ;;
            lh-aloha) available=(top right_wrist) ;;
            *) available=(image) ;;
        esac
        lte_keys="$(IFS=,; echo "${available[*]:0:LTE_IMAGE_COUNT}")"
    fi
    for ((i=0; i<TRAIN_SEQUENTIAL_RUNS; i++)); do
        if [[ "${TRAINING_METHOD}" == lte ]]; then
            run_name="${TARGET_ENVIRONMENT//-/_}_lte_${LTE_ARCHITECTURE}_${timestamp}"
        else
            run_name="${TARGET_ENVIRONMENT//-/_}_ptp_${timestamp}"
        fi
        run_name="$(next_available_run_name "${output_root}" "${run_name}")"
        output_dir="${output_root}/${run_name}"
        if [[ -n "${RUN_NOTE}" ]]; then
            mkdir -p "${output_dir}"
            printf '%s\n' "${RUN_NOTE}" >"${output_dir}/note.txt"
        fi
        overrides=("training.seed=$((TRAIN_SEED + i))" "training.device=cuda:0" "dataloader.batch_size=${TRAIN_BATCH_SIZE}" "val_dataloader.batch_size=${TRAIN_BATCH_SIZE}" "dataloader.num_workers=${TRAIN_DATALOADER_WORKERS}" "val_dataloader.num_workers=${TRAIN_DATALOADER_WORKERS}" "dataloader.pin_memory=${TRAIN_PIN_MEMORY}" "val_dataloader.pin_memory=${TRAIN_PIN_MEMORY}" "dataloader.persistent_workers=${TRAIN_PERSISTENT_WORKERS}" "val_dataloader.persistent_workers=${TRAIN_PERSISTENT_WORKERS}" "+dataloader.prefetch_factor=${TRAIN_PREFETCH_FACTOR}" "+val_dataloader.prefetch_factor=${TRAIN_PREFETCH_FACTOR}" "+task.dataset.image_augmentation=${TRAIN_IMAGE_AUGMENTATION}" "+task.dataset.cache_images_on_gpu=${TRAIN_CACHE_IMAGES_ON_GPU}")
        [[ "${TRAIN_EPOCHS}" == "auto" ]] || overrides+=("training.num_epochs=${TRAIN_EPOCHS}")
        log "Starting ${TRAINING_METHOD} training: ${run_name}"
        if [[ "${TRAINING_METHOD}" == lte ]]; then
            [[ "${LTE_ARCHITECTURE}" == transformer ]] && overrides+=("optimizer.learning_rate=${TRAIN_LEARNING_RATE}") || overrides+=("optimizer.lr=${TRAIN_LEARNING_RATE}")
            overrides+=("policy.temporal_rgb_keys=[${lte_keys}]" "policy.temporal_multi_image_fusion_enabled=$([[ "${lte_keys}" == *,* ]] && echo true || echo false)" "policy.temporal_embedding_cache_enabled=${LTE_EMBEDDING_CACHE}" "policy.temporal_embedding_cache_start_epoch=${LTE_CACHE_START_EPOCH}" "policy.temporal_embedding_cache_warmup_epochs=${LTE_CACHE_WARMUP_EPOCHS}" "policy.temporal_embedding_cache_refresh_epochs=${LTE_CACHE_REFRESH_EPOCHS}" "policy.history_reconstruction.num_history_queries=${LTE_HISTORY_DECODER_SAMPLES}" "policy.temporal_latent_dim=${LTE_TEMPORAL_LATENT_DIM}" "policy.temporal_hidden_dim=${LTE_TEMPORAL_HIDDEN_DIM}" "policy.temporal_num_hidden_layers=${LTE_TEMPORAL_HIDDEN_LAYERS}" "policy.history_reconstruction.hidden_dim=${LTE_HISTORY_DECODER_HIDDEN_DIM}" "policy.history_reconstruction.num_hidden_layers=${LTE_HISTORY_DECODER_HIDDEN_LAYERS}")
            CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" MUJOCO_GL=egl SDL_VIDEODRIVER=dummy ./train_lte_img_not.sh "${train_task}" "${output_dir}" "${overrides[@]}"
        else
            case "${TARGET_ENVIRONMENT}" in
                square) config_dir=experiment_configs/square; config_name=transformer_square_paper ;;
                tool-hang) config_dir=experiment_configs/tool; config_name=transformer_tool_hang_paper ;;
                transport) config_dir=experiment_configs/transport; config_name=transformer_transport_paper ;;
                lh-aloha) config_dir=experiment_configs/aloha; config_name=transformer_aloha_paper ;;
                lh-square) config_dir=experiment_configs/longhist; config_name=transformer_longhist_paper ;;
                *) echo "PTP is unsupported for ${TARGET_ENVIRONMENT}." >&2; return 1 ;;
            esac
            CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" MUJOCO_GL=egl SDL_VIDEODRIVER=dummy python train.py --config-dir="${config_dir}" --config-name="${config_name}" hydra.run.dir="${output_dir}" "logging.name=${run_name}" logging.id=null "optimizer.learning_rate=${TRAIN_LEARNING_RATE}" "${overrides[@]}"
        fi
    done
}

run_training
[[ -n "${NTFY_AUTH_TOKEN:-}" ]] && curl --fail --silent --show-error --max-time 20 \
    -H "Authorization: Bearer ${NTFY_AUTH_TOKEN}" -H "Title: LDP training complete" -H "Tags: white_check_mark" \
    --data-binary "LDP training complete on $(hostname): env=${TARGET_ENVIRONMENT}, method=${TRAINING_METHOD}, runs=${TRAIN_SEQUENTIAL_RUNS}, output=${EXTERNAL_OUTPUT_ROOT:-${DATA_ROOT}/outputs}" \
    "${NTFY_SERVER%/}/${NTFY_TOPIC}" || true
[[ "${SHUTDOWN_AFTER_COMPLETION}" == "true" ]] && as_root shutdown -h now || true
