#!/usr/bin/env bash
# Fresh Ubuntu VM setup for LDP.
#
# Before running, edit only the CONFIGURATION block below. Do not commit a
# copy containing real tokens or private-key paths.
#
# Run from a copied file on the new VM:
#   bash setup_fresh_vm.sh

set -Eeuo pipefail
umask 077

###############################################################################
# CONFIGURATION -- edit these values before running on the new VM.
###############################################################################

# Repository and install location. HTTPS works for public repositories. For a
# private HTTPS repository, set GITHUB_TOKEN; it is used only during clone and
# never written into the Git remote URL.
REPO_URL="https://github.com/alek5k/ldp.git"
REPO_DIR="$HOME/Repos/ldp"
GITHUB_TOKEN=""                    # Fine-grained token with repository read access.

# Alternatively use an SSH URL such as git@github.com:OWNER/ldp.git and point
# this at an existing private key. Leave empty to use your usual SSH agent.
GITHUB_SSH_KEY_FILE=""              # e.g. "$HOME/.ssh/id_ed25519"

# W&B authentication. Leave blank to skip login and run `wandb login` later.
WANDB_API_KEY=""
WANDB_ENTITY="uts_robot_lab"        # Default entity exported on Conda activation.

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
conda config --set channel_priority strict
conda config --set solver libmamba || true

if [[ -n "${GITHUB_SSH_KEY_FILE}" ]]; then
    if [[ ! -r "${GITHUB_SSH_KEY_FILE}" ]]; then
        echo "GITHUB_SSH_KEY_FILE is not readable: ${GITHUB_SSH_KEY_FILE}" >&2
        exit 1
    fi
    export GIT_SSH_COMMAND="ssh -i ${GITHUB_SSH_KEY_FILE} -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
fi

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
    if [[ -n "${GITHUB_TOKEN}" && "${REPO_URL}" == https://github.com/* ]]; then
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
    fi
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

if [[ -n "${WANDB_API_KEY}" ]]; then
    log "Logging into Weights & Biases"
    export WANDB_API_KEY WANDB_ENTITY
    conda run -n "${CONDA_ENV_NAME}" python - <<'PY'
import os
import wandb

wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
PY
else
    warn "WANDB_API_KEY is blank; run 'conda activate ${CONDA_ENV_NAME} && wandb login' later."
fi

log "Verifying the environment"
conda run -n "${CONDA_ENV_NAME}" python - <<'PY'
import torch
import wandb
import diffusion_policy

print(f"Python setup OK; torch={torch.__version__}; CUDA available={torch.cuda.is_available()}; wandb={wandb.__version__}")
PY

cat <<EOF

Setup complete.

Next shell:
  source "${CONDA_DIR}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
  cd "${REPO_DIR}"
  python experiment_cli.py
EOF
