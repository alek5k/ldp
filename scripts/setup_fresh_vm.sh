#!/usr/bin/env bash
# Fresh Ubuntu VM setup for LDP.
#
# Before running, edit the QUICK CONFIGURATION block below. Do not commit a
# copy containing real tokens or private-key paths.
#
# Run from a copied file on the new VM:
#   bash setup_fresh_vm.sh

set -Eeuo pipefail
umask 077

###############################################################################
# QUICK CONFIGURATION -- normally the only section you need to edit.
# Do not commit a copy containing real tokens or private keys.
###############################################################################

# Dataset to fetch from the links documented in README.md. Supported values:
#   none | square | tool-hang | transport | lh-aloha | lh-square
# square, tool-hang, and transport share the Robomimic image archive.
# WaitAtGoal, LiftQA, PianoSim, and PushT have no public download link in this
# README, so copy those datasets to DATA_ROOT separately.
TARGET_ENVIRONMENT="lh-aloha"
TRAINING_METHOD="lte"                # lte | ptp (PTP supports lh-aloha/lh-square)

# GitHub / W&B credentials. Use a fine-grained GitHub token with repository
# Contents: Read access. It is supplied through a temporary askpass helper and
# never written into the Git remote URL or Git configuration.
GITHUB_TOKEN=""
WANDB_API_KEY=""
NTFY_AUTH_TOKEN=''

TRAIN_GPU="0"                        # Physical GPU index passed via CUDA_VISIBLE_DEVICES.
TRAIN_EPOCHS=500
TRAIN_BATCH_SIZE=64
TRAIN_LEARNING_RATE="0.0001"
TRAIN_SEED=42
TRAIN_SEQUENTIAL_RUNS=1

# Requires sudo. Leave false for setup-only machines. `true` shuts down only
# after the final requested training run exits successfully.
SHUTDOWN_AFTER_SUCCESS=true
SHUTDOWN_DELAY_MINUTES=0

###############################################################################
# MACHINE CONFIGURATION -- change only when the VM layout differs.
###############################################################################
TRAIN_RUN_NAME=""                    # Blank: generate a timestamped name.
TRAIN_OUTPUT_ROOT=""                 # Blank: external root if configured, else DATA_ROOT/outputs.
TRAIN_CHECKPOINT_EVERY=""            # Blank: keep the selected config's value.
TRAIN_ROLLOUT_EVERY=""               # Blank: keep the selected config's value.

RUN_TRAINING=true
DOWNLOAD_OBS_ENCODERS=false          # Also fetch the optional embedding encoders. (PTP)

NTFY_SERVER=https://ntfy.aleksk.net
NTFY_TOPIC="ldp"                     # Change if your ntfy token uses another topic.

# Repository and install location.
REPO_URL="https://github.com/alek5k/ldp.git"
REPO_DIR="$HOME/ldp"

# W&B organisation exported whenever the environment is activated.
WANDB_ENTITY="uts_robot_lab"

# Conda / Python environment.
CONDA_DIR="$HOME/miniforge3"
CONDA_ENV_NAME="robodiff-lh-5090"
CONDA_ENV_FILE="conda_environment.yaml"

# Data locations. Put DATA_ROOT on a large mounted disk if desired. The script
# makes REPO_DIR/data a symlink when it differs from DATA_ROOT.
DATA_ROOT="$REPO_DIR/data"
EXTERNAL_OUTPUT_ROOT=""             # Optional, e.g. /mnt/wdblack/ldp/data/outputs

# System setup. NVIDIA drivers are intentionally not installed: use the driver
# supplied by your VM image / cloud provider, then verify it with nvidia-smi.
INSTALL_SYSTEM_PACKAGES=true
INSTALL_GIT_LFS=true

###############################################################################
# End configuration.
###############################################################################

readonly MINIFORGE_VERSION="24.11.3-0"
readonly MINIFORGE_BASE_URL="https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}"

log() {
    printf '\n==> %s\n' "$*"
}

warn() {
    printf '\nWARNING: %s\n' "$*" >&2
}

as_root() {
    if [[ "${EUID}" -eq 0 ]]; then
        "$@"
    else
        sudo "$@"
    fi
}

cleanup() {
    if [[ -n "${GIT_ASKPASS_FILE:-}" && -e "${GIT_ASKPASS_FILE}" ]]; then
        rm -f "${GIT_ASKPASS_FILE}"
    fi
}
trap cleanup EXIT

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

training_output_root() {
    if [[ -n "${TRAIN_OUTPUT_ROOT}" ]]; then
        printf '%s\n' "${TRAIN_OUTPUT_ROOT}"
    elif [[ -n "${EXTERNAL_OUTPUT_ROOT}" ]]; then
        printf '%s\n' "${EXTERNAL_OUTPUT_ROOT}"
    else
        printf '%s\n' "${DATA_ROOT}/outputs"
    fi
}

run_training() {
    local output_root base_name run_name output_dir config_dir config_name train_task
    local -a overrides

    if [[ "${RUN_TRAINING}" != "true" ]]; then
        return
    fi
    if [[ "${TARGET_ENVIRONMENT}" == "none" || -z "${TARGET_ENVIRONMENT}" ]]; then
        echo "Set TARGET_ENVIRONMENT before setting RUN_TRAINING=true." >&2
        exit 1
    fi
    if [[ "${TRAINING_METHOD}" != "lte" && "${TRAINING_METHOD}" != "ptp" ]]; then
        echo "TRAINING_METHOD must be 'lte' or 'ptp'." >&2
        exit 1
    fi
    if ! [[ "${TRAIN_GPU}" =~ ^[0-9]+$ && "${TRAIN_EPOCHS}" =~ ^[1-9][0-9]*$ \
        && "${TRAIN_BATCH_SIZE}" =~ ^[1-9][0-9]*$ && "${TRAIN_SEED}" =~ ^[0-9]+$ \
        && "${TRAIN_SEQUENTIAL_RUNS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "TRAIN_GPU, TRAIN_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_SEED, and TRAIN_SEQUENTIAL_RUNS must be valid integers." >&2
        exit 1
    fi

    output_root="$(training_output_root)"
    mkdir -p "${output_root}"
    base_name="${TRAIN_RUN_NAME:-${TARGET_ENVIRONMENT//-/_}_${TRAINING_METHOD}_$(date +%Y%m%d_%H%M%S)}"
    train_task="${TARGET_ENVIRONMENT}"
    if [[ "${train_task}" == "tool-hang" ]]; then
        train_task="tool_hang"
    fi

    for ((run_index = 0; run_index < TRAIN_SEQUENTIAL_RUNS; run_index++)); do
        run_name="${base_name}"
        if (( TRAIN_SEQUENTIAL_RUNS > 1 )); then
            run_name+="-r$((run_index + 1))"
        fi
        output_dir="${output_root}/${run_name}"
        if [[ -e "${output_dir}" ]]; then
            echo "Training output directory already exists: ${output_dir}" >&2
            exit 1
        fi

        overrides=(
            "training.num_epochs=${TRAIN_EPOCHS}"
            "training.seed=$((TRAIN_SEED + run_index))"
            "training.device=cuda:0"
            "dataloader.batch_size=${TRAIN_BATCH_SIZE}"
            "val_dataloader.batch_size=${TRAIN_BATCH_SIZE}"
            "logging.name=${run_name}"
        )
        if [[ -n "${TRAIN_CHECKPOINT_EVERY}" ]]; then
            overrides+=("training.checkpoint_every=${TRAIN_CHECKPOINT_EVERY}")
        fi
        if [[ -n "${TRAIN_ROLLOUT_EVERY}" ]]; then
            overrides+=("training.rollout_every=${TRAIN_ROLLOUT_EVERY}")
        fi

        log "Starting ${TRAINING_METHOD} training: ${run_name}"
        case "${TRAINING_METHOD}" in
            lte)
                overrides+=("optimizer.lr=${TRAIN_LEARNING_RATE}")
                (
                    cd "${REPO_DIR}"
                    CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" MUJOCO_GL=egl SDL_VIDEODRIVER=dummy \
                        ./train_lte_img_not.sh "${train_task}" "${output_dir}" "${overrides[@]}"
                )
                ;;
            ptp)
                case "${TARGET_ENVIRONMENT}" in
                    lh-aloha) config_dir="experiment_configs/aloha"; config_name="transformer_aloha" ;;
                    lh-square) config_dir="experiment_configs/longhist"; config_name="transformer_longhist" ;;
                    *)
                        echo "PTP only supports TARGET_ENVIRONMENT=lh-aloha or lh-square." >&2
                        exit 1
                        ;;
                esac
                overrides+=("optimizer.learning_rate=${TRAIN_LEARNING_RATE}")
                (
                    cd "${REPO_DIR}"
                    CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" MUJOCO_GL=egl SDL_VIDEODRIVER=dummy \
                        python train.py --config-dir="${config_dir}" --config-name="${config_name}" \
                        hydra.run.dir="${output_dir}" "${overrides[@]}"
                )
                ;;
        esac
    done
}

notify_training_complete() {
    local message

    if [[ "${RUN_TRAINING}" != "true" ]]; then
        return
    fi
    if [[ -z "${NTFY_AUTH_TOKEN}" ]]; then
        warn "NTFY_AUTH_TOKEN is blank; skipping training-complete notification."
        return
    fi
    if ! command -v curl >/dev/null 2>&1; then
        warn "curl is unavailable; skipping training-complete notification."
        return
    fi

    message="LDP training complete on $(hostname): env=${TARGET_ENVIRONMENT}, method=${TRAINING_METHOD}, runs=${TRAIN_SEQUENTIAL_RUNS}, output=$(training_output_root)"
    if ! curl --fail --silent --show-error --max-time 20 \
        -H "Authorization: Bearer ${NTFY_AUTH_TOKEN}" \
        -H "Title: LDP training complete" \
        -H "Tags: white_check_mark" \
        --data-binary "${message}" \
        "${NTFY_SERVER%/}/${NTFY_TOPIC}"; then
        warn "Could not send the ntfy completion notification."
    fi
}

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "This script currently supports Ubuntu/Debian Linux only." >&2
    exit 1
fi

case "$(uname -m)" in
    x86_64) MINIFORGE_ARCH="x86_64" ;;
    aarch64|arm64) MINIFORGE_ARCH="aarch64" ;;
    *) echo "Unsupported CPU architecture: $(uname -m)" >&2; exit 1 ;;
esac

if [[ "${INSTALL_SYSTEM_PACKAGES}" == "true" ]]; then
    log "Installing Ubuntu system packages"
    as_root apt-get update
    as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential ca-certificates cmake curl ffmpeg git git-lfs \
        libegl1 libgl1 libglib2.0-0 libglfw3 libglew-dev libosmesa6-dev \
        patchelf pkg-config rsync screen unzip wget
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    log "Detected NVIDIA driver"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    warn "nvidia-smi is unavailable. Install a compatible NVIDIA driver before GPU training."
fi

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
# Pytorch3D comes from its own channel while AV/FFmpeg comes from Conda Forge;
# flexible priority lets the solver combine those compatible packages.
conda config --set channel_priority flexible
conda config --set solver libmamba || true

configure_github_token_auth

if [[ -d "${REPO_DIR}/.git" ]]; then
    log "Updating existing checkout at ${REPO_DIR}"
    git -C "${REPO_DIR}" fetch --tags origin
    git -C "${REPO_DIR}" pull --ff-only
else
    if [[ -e "${REPO_DIR}" ]]; then
        echo "REPO_DIR already exists but is not a Git checkout: ${REPO_DIR}" >&2
        exit 1
    fi

    log "Cloning ${REPO_URL}"
    mkdir -p "$(dirname "${REPO_DIR}")"
    git clone "${REPO_URL}" "${REPO_DIR}"
fi

git -C "${REPO_DIR}" submodule update --init --recursive
if [[ "${INSTALL_GIT_LFS}" == "true" ]] && command -v git-lfs >/dev/null 2>&1; then
    git -C "${REPO_DIR}" lfs install --local
    git -C "${REPO_DIR}" lfs pull
fi

ENV_FILE_PATH="${REPO_DIR}/${CONDA_ENV_FILE}"
if [[ ! -f "${ENV_FILE_PATH}" ]]; then
    echo "Conda environment file not found: ${ENV_FILE_PATH}" >&2
    exit 1
fi

if conda env list | awk '{print $1}' | grep -Fxq "${CONDA_ENV_NAME}"; then
    log "Updating Conda environment ${CONDA_ENV_NAME}"
    conda env update --prune -n "${CONDA_ENV_NAME}" -f "${ENV_FILE_PATH}"
else
    log "Creating Conda environment ${CONDA_ENV_NAME}"
    conda env create -n "${CONDA_ENV_NAME}" -f "${ENV_FILE_PATH}"
fi

log "Installing LDP package"
conda run -n "${CONDA_ENV_NAME}" python -m pip install --upgrade pip setuptools wheel
conda run -n "${CONDA_ENV_NAME}" python -m pip install -e "${REPO_DIR}"

# Keep the chosen entity available to future `conda activate` shells without
# storing the W&B API key in a project file.
ENV_PREFIX="${CONDA_DIR}/envs/${CONDA_ENV_NAME}"
mkdir -p "${ENV_PREFIX}/etc/conda/activate.d"
cat >"${ENV_PREFIX}/etc/conda/activate.d/ldp-wandb.sh" <<EOF
export WANDB_ENTITY="${WANDB_ENTITY}"
EOF
conda activate "${CONDA_ENV_NAME}"

# requirements.txt is an exported environment snapshot containing paths from
# the original machine. conda_environment.yaml above is the portable source.

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

if [[ -n "${WANDB_API_KEY}" ]]; then
    log "Logging into Weights & Biases"
    export WANDB_API_KEY WANDB_ENTITY
    python -c 'import os, wandb; wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)'
else
    warn "WANDB_API_KEY is blank; run 'conda activate ${CONDA_ENV_NAME} && wandb login' later."
fi

log "Verifying the environment"
python -c 'import torch, wandb, diffusion_policy; print(f"Python setup OK; torch={torch.__version__}; CUDA available={torch.cuda.is_available()}; wandb={wandb.__version__}")'

run_training
notify_training_complete

if [[ "${SHUTDOWN_AFTER_SUCCESS}" == "true" ]]; then
    if [[ "${RUN_TRAINING}" != "true" ]]; then
        echo "SHUTDOWN_AFTER_SUCCESS=true requires RUN_TRAINING=true." >&2
        exit 1
    fi
    if ! [[ "${SHUTDOWN_DELAY_MINUTES}" =~ ^[0-9]+$ ]]; then
        echo "SHUTDOWN_DELAY_MINUTES must be a non-negative integer." >&2
        exit 1
    fi
    log "All requested training completed successfully; shutting down in ${SHUTDOWN_DELAY_MINUTES} minute(s)"
    as_root shutdown -h "+${SHUTDOWN_DELAY_MINUTES}"
fi

cat <<EOF

Setup complete.

Next shell:
  source "${CONDA_DIR}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
  cd "${REPO_DIR}"
  python experiment_cli.py
EOF
