#!/usr/bin/env bash
# Fresh Ubuntu VM setup for LDP. Edit the QUICK CONFIGURATION block, then run
# this copied script on the new VM.

set -Eeuo pipefail
umask 077

###############################################################################
# QUICK CONFIGURATION -- normally the only section you need to edit.
###############################################################################

# Auth Tokens
GITHUB_TOKEN=""
WANDB_API_KEY=""
NTFY_AUTH_TOKEN=""

TARGET_ENVIRONMENT="lh-aloha"         # square | transport | tool-hang | lh-aloha | lh-square | waitatgoal | liftqa
TRAINING_METHOD="lte"                 # lte | ptp
LTE_ARCHITECTURE="unet"               # transformer | unet
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


TRAIN_GPU="0"
TRAIN_EPOCHS=5
TRAIN_BATCH_SIZE=64
TRAIN_LEARNING_RATE="0.0001"
TRAIN_SEED=42
TRAIN_SEQUENTIAL_RUNS=1
DOWNLOAD_OBS_ENCODERS=false

SHUTDOWN_AFTER_COMPLETION=false

###############################################################################
# MACHINE CONFIGURATION -- normally leave unchanged.
###############################################################################

REPO_URL="https://github.com/alek5k/ldp.git"
REPO_DIR="$HOME/ldp"
WANDB_ENTITY="uts_robot_lab"
NTFY_SERVER="https://ntfy.aleksk.net"
NTFY_TOPIC="ldp"
CONDA_DIR="$HOME/miniforge3"
CONDA_ENV_NAME="robodiff-lh-5090"
CONDA_ENV_FILE="conda_environment.yaml"
DATA_ROOT="$REPO_DIR/data"
EXTERNAL_OUTPUT_ROOT=""

###############################################################################

readonly MINIFORGE_VERSION="24.11.3-0"
readonly MINIFORGE_BASE_URL="https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}"

log() { printf '\n==> %s\n' "$*"; }
warn() { printf '\nWARNING: %s\n' "$*" >&2; }
as_root() { if [[ "${EUID}" -eq 0 ]]; then "$@"; else sudo "$@"; fi; }

on_exit() {
    local exit_status=$?
    trap - EXIT
    set +e
    [[ -n "${GIT_ASKPASS_FILE:-}" ]] && rm -f "${GIT_ASKPASS_FILE}" || true
    if (( exit_status != 0 )); then
        [[ -n "${NTFY_AUTH_TOKEN}" ]] && curl --fail --silent --show-error --max-time 20 \
            -H "Authorization: Bearer ${NTFY_AUTH_TOKEN}" -H "Title: LDP setup/training failed" -H "Tags: warning" \
            --data-binary "LDP setup/training failed on $(hostname): exit_status=${exit_status}; env=${TARGET_ENVIRONMENT}; method=${TRAINING_METHOD}" \
            "${NTFY_SERVER%/}/${NTFY_TOPIC}" || true
        if [[ "${SHUTDOWN_AFTER_COMPLETION}" == "true" ]]; then
            as_root shutdown -h now || true
        fi
    fi
    exit "${exit_status}"
}
trap on_exit EXIT

configure_github_token_auth() {
    if [[ -z "${GITHUB_TOKEN}" || "${REPO_URL}" != https://github.com/* ]]; then
        return
    fi

    GIT_ASKPASS_FILE="$(mktemp git-askpass-XXXXXX)"
    cat >"${GIT_ASKPASS_FILE}" <<'EOF'
#!/usr/bin/env bash
case "$1" in
    *Username*) printf '%s\n' 'x-access-token' ;;
    *) printf '%s\n' "${GITHUB_TOKEN}" ;;
esac
EOF
    chmod 700 "${GIT_ASKPASS_FILE}"
    export GIT_ASKPASS="${GIT_ASKPASS_FILE}"
    export GIT_TERMINAL_PROMPT=0
    export GITHUB_TOKEN
}

download_google_drive_zip() {
    local file_id="$1"
    local archive_name="$2"
    local expected_path="$3"
    local extract_root="${4:-${DATA_ROOT}}"
    local archive_path="${extract_root}/${archive_name}"
    local cookie_file html_file uuid confirm final_url

    if [[ -e "${expected_path}" ]]; then
        log "Dataset already present: ${expected_path}"
        return
    fi

    log "Downloading ${archive_name} from Google Drive"
    mkdir -p "${extract_root}"
    cookie_file="$(mktemp)"
    html_file="$(mktemp)"
    wget --quiet --save-cookies "${cookie_file}" --keep-session-cookies \
        "https://drive.google.com/uc?export=download&id=${file_id}" \
        -O "${html_file}"
    uuid="$(grep -oP 'name="uuid" value="\K[^"]+' "${html_file}" || true)"
    confirm="$(grep -oP 'name="confirm" value="\K[^"]+' "${html_file}" || true)"
    if [[ -z "${uuid}" || -z "${confirm}" ]]; then
        rm -f "${cookie_file}" "${html_file}"
        echo "Google Drive confirmation changed or requires a browser. Download manually from README.md." >&2
        exit 1
    fi
    final_url="https://drive.usercontent.google.com/download?id=${file_id}&export=download&confirm=${confirm}&uuid=${uuid}"
    wget --show-progress --load-cookies "${cookie_file}" "${final_url}" -O "${archive_path}"
    rm -f "${cookie_file}" "${html_file}"
    unzip -q -o "${archive_path}" -d "${extract_root}"
    rm -f "${archive_path}"
}

download_observation_encoders() {
    if [[ "${DOWNLOAD_OBS_ENCODERS}" == "true" ]]; then
        download_google_drive_zip \
            "1tSYyWg3HZbTtEhzpAXQpl28DSrWsXc7J" \
            "obs_encoders.zip" \
            "${REPO_DIR}/obs_encoders" \
            "${REPO_DIR}"
    fi
}

download_selected_dataset() {
    case "${TARGET_ENVIRONMENT}" in
        none|"")
            log "Skipping dataset download (TARGET_ENVIRONMENT=none)"
            ;;
        square)
            download_robomimic_image "${DATA_ROOT}/robomimic/datasets/square/mh/image_abs.hdf5"
            ;;
        tool-hang|tool_hang)
            download_robomimic_image "${DATA_ROOT}/robomimic/datasets/tool_hang/ph/image_abs.hdf5"
            ;;
        transport)
            download_robomimic_image "${DATA_ROOT}/robomimic/datasets/transport/mh/image_abs.hdf5"
            ;;
        lh-aloha)
            download_google_drive_zip \
                "1gwzIRBmn0a4Orj2okMNQ9qiPPpxmqdKA" \
                "aloha_twomodes_single.zip" \
                "${DATA_ROOT}/aloha_twomodes_single/demos.hdf5"
            ;;
        lh-square)
            download_google_drive_zip \
                "1-ZDi8-aVx1I8aZCan-vXJQIpLyCCNwym" \
                "longhistsquare100.zip" \
                "${DATA_ROOT}/longhistsquare100/demos.hdf5"
            ;;
        waitatgoal|liftqa|pianosim|pusht)
            warn "${TARGET_ENVIRONMENT} has no public dataset download in README.md; expecting an existing dataset under ${DATA_ROOT}."
            ;;
        *)
            echo "Unknown TARGET_ENVIRONMENT: ${TARGET_ENVIRONMENT}" >&2
            echo "Choose: none, square, tool-hang, transport, lh-aloha, lh-square" >&2
            exit 1
            ;;
    esac
}

download_robomimic_image() {
    local expected_path="$1"
    local archive_path="${DATA_ROOT}/robomimic_image.zip"

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

run_training() {
    local output_root run_name output_dir train_task lte_keys
    local -a overrides available
    output_root="${EXTERNAL_OUTPUT_ROOT:-${DATA_ROOT}/outputs}"
    mkdir -p "${output_root}"
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
    for ((i=0;i<TRAIN_SEQUENTIAL_RUNS;i++)); do
        run_name="${TARGET_ENVIRONMENT//-/_}_${TRAINING_METHOD}_$(date +%Y%m%d_%H%M%S)"
        (( TRAIN_SEQUENTIAL_RUNS > 1 )) && run_name+="-r$((i+1))"
        output_dir="${output_root}/${run_name}"
        overrides=("training.num_epochs=${TRAIN_EPOCHS}" "training.seed=$((TRAIN_SEED+i))" "training.device=cuda:0" "dataloader.batch_size=${TRAIN_BATCH_SIZE}" "val_dataloader.batch_size=${TRAIN_BATCH_SIZE}" "logging.name=${run_name}")
        log "Starting ${TRAINING_METHOD} training: ${run_name}"
        if [[ "${TRAINING_METHOD}" == lte ]]; then
            [[ "${LTE_ARCHITECTURE}" == transformer ]] && overrides+=("optimizer.learning_rate=${TRAIN_LEARNING_RATE}") || overrides+=("optimizer.lr=${TRAIN_LEARNING_RATE}")
            overrides+=("policy.temporal_rgb_keys=[${lte_keys}]" "policy.temporal_multi_image_fusion_enabled=$([[ "${lte_keys}" == *,* ]] && echo true || echo false)" "policy.temporal_embedding_cache_enabled=${LTE_EMBEDDING_CACHE}" "policy.temporal_embedding_cache_start_epoch=${LTE_CACHE_START_EPOCH}" "policy.temporal_embedding_cache_warmup_epochs=${LTE_CACHE_WARMUP_EPOCHS}" "policy.temporal_embedding_cache_refresh_epochs=${LTE_CACHE_REFRESH_EPOCHS}" "policy.history_reconstruction.num_history_queries=${LTE_HISTORY_DECODER_SAMPLES}" "policy.temporal_latent_dim=${LTE_TEMPORAL_LATENT_DIM}" "policy.temporal_hidden_dim=${LTE_TEMPORAL_HIDDEN_DIM}" "policy.temporal_num_hidden_layers=${LTE_TEMPORAL_HIDDEN_LAYERS}" "policy.history_reconstruction.hidden_dim=${LTE_HISTORY_DECODER_HIDDEN_DIM}" "policy.history_reconstruction.num_hidden_layers=${LTE_HISTORY_DECODER_HIDDEN_LAYERS}")
            (cd "${REPO_DIR}"; CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" MUJOCO_GL=egl SDL_VIDEODRIVER=dummy ./train_lte_img_not.sh "${train_task}" "${output_dir}" "${overrides[@]}")
        else
            case "${TARGET_ENVIRONMENT}" in lh-aloha) config_dir=experiment_configs/aloha; config_name=transformer_aloha ;; *) config_dir=experiment_configs/longhist; config_name=transformer_longhist ;; esac
            (cd "${REPO_DIR}"; CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" MUJOCO_GL=egl SDL_VIDEODRIVER=dummy python train.py --config-dir="${config_dir}" --config-name="${config_name}" hydra.run.dir="${output_dir}" "optimizer.learning_rate=${TRAIN_LEARNING_RATE}" "${overrides[@]}")
        fi
    done
}

log "Installing system packages"
as_root apt-get update
as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential ca-certificates cmake curl ffmpeg git git-lfs \
    libegl1 libgl1 libglib2.0-0 libglfw3 libglew-dev libosmesa6-dev \
    patchelf pkg-config rsync screen unzip wget
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
MINIFORGE_ARCH="x86_64"

if [[ ! -x "${CONDA_DIR}/bin/conda" ]]; then
    log "Installing Miniforge in ${CONDA_DIR}"
    INSTALLER="$(mktemp --suffix=.sh miniforge-XXXXXX)"
    curl --fail --location --retry 3 \
        "${MINIFORGE_BASE_URL}/Miniforge3-${MINIFORGE_VERSION}-Linux-${MINIFORGE_ARCH}.sh" \
        --output "${INSTALLER}"
    bash "${INSTALLER}" -b -p "${CONDA_DIR}"
    rm -f "${INSTALLER}"
fi

source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda config --set channel_priority flexible
conda config --set solver libmamba || true

configure_github_token_auth

if [[ -d "${REPO_DIR}/.git" ]]; then
    log "Updating existing checkout at ${REPO_DIR}"
    git -C "${REPO_DIR}" pull --ff-only
else
    log "Cloning ${REPO_URL}"
    mkdir -p "$(dirname "${REPO_DIR}")"
    git clone "${REPO_URL}" "${REPO_DIR}"
fi

git -C "${REPO_DIR}" submodule update --init --recursive
git -C "${REPO_DIR}" lfs install --local
git -C "${REPO_DIR}" lfs pull
cd "${REPO_DIR}"

ENV_FILE_PATH="${REPO_DIR}/${CONDA_ENV_FILE}"

if conda env list | awk '{print $1}' | grep -Fxq "${CONDA_ENV_NAME}"; then
    log "Updating Conda environment ${CONDA_ENV_NAME}"
    conda env update --prune -n "${CONDA_ENV_NAME}" -f "${ENV_FILE_PATH}"
else
    log "Creating Conda environment ${CONDA_ENV_NAME}"
    conda env create -n "${CONDA_ENV_NAME}" -f "${ENV_FILE_PATH}"
fi

log "Installing LDP package"
conda activate "${CONDA_ENV_NAME}"
python -m pip install --upgrade pip 'setuptools<81' wheel
python -m pip install -e .

if [[ "${DATA_ROOT}" != "${REPO_DIR}/data" ]]; then
    mkdir -p "${DATA_ROOT}"
    if [[ -e "${REPO_DIR}/data" && ! -L "${REPO_DIR}/data" ]]; then
        echo "${REPO_DIR}/data exists and is not a symlink; refusing to replace it." >&2
        exit 1
    fi
    ln -sfn "${DATA_ROOT}" "${REPO_DIR}/data"
else
    mkdir -p "${DATA_ROOT}"
fi
mkdir -p "${DATA_ROOT}/outputs" "${DATA_ROOT}/inference" "${DATA_ROOT}/videos"

if [[ -n "${EXTERNAL_OUTPUT_ROOT}" ]]; then
    mkdir -p "${EXTERNAL_OUTPUT_ROOT}"
    ln -sfn "${EXTERNAL_OUTPUT_ROOT}" "${DATA_ROOT}/outputs_extdrive"
fi

download_selected_dataset
download_observation_encoders

log "Logging into Weights & Biases"
export WANDB_API_KEY WANDB_ENTITY
python -c 'import os, wandb; wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)'

log "Verifying the environment"
python -c 'import torch, wandb, diffusion_policy; print(f"Python setup OK; torch={torch.__version__}; CUDA available={torch.cuda.is_available()}; wandb={wandb.__version__}")'

run_training
[[ -n "${NTFY_AUTH_TOKEN}" ]] && curl --fail --silent --show-error --max-time 20 \
    -H "Authorization: Bearer ${NTFY_AUTH_TOKEN}" -H "Title: LDP training complete" -H "Tags: white_check_mark" \
    --data-binary "LDP training complete on $(hostname): env=${TARGET_ENVIRONMENT}, method=${TRAINING_METHOD}, runs=${TRAIN_SEQUENTIAL_RUNS}, output=${EXTERNAL_OUTPUT_ROOT:-${DATA_ROOT}/outputs}" \
    "${NTFY_SERVER%/}/${NTFY_TOPIC}" || true
if [[ "${SHUTDOWN_AFTER_COMPLETION}" == "true" ]]; then
    as_root shutdown -h now || true
fi
