#!/usr/bin/env bash
# Fresh Ubuntu VM setup for LDP. Edit the QUICK CONFIGURATION block, then run
# this copied script on the new VM.

set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Keep credentials out of the setup script and repository. An absolute or
# alternate path can be supplied with LDP_VM_CREDENTIALS_FILE.
CREDENTIALS_FILE="${LDP_VM_CREDENTIALS_FILE:-${SCRIPT_DIR}/vm_credentials.sh}"

###############################################################################
# QUICK CONFIGURATION -- normally the only section you need to edit.
###############################################################################

RUN_SETUP=true                        # false: reuse the existing checkout and Conda environment.

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
# Keep large archives and extracted datasets off the VM boot disk. The setup
# below makes ~/ldp/data point here, so existing relative config paths work.
# Empty chooses the candidate filesystem with the most available space.
# Set explicitly only to override that choice.
DATA_ROOT=""
EXTERNAL_OUTPUT_ROOT=""

###############################################################################

if [[ -r "${CREDENTIALS_FILE}" ]]; then
    # shellcheck source=/dev/null
    source "${CREDENTIALS_FILE}"
elif [[ -z "${WANDB_API_KEY:-}" ]]; then
    cat >&2 <<EOF
Missing VM credentials: ${CREDENTIALS_FILE}
Copy scripts/vm_credentials.sh.example to that path, fill in the values, and
run chmod 600 ${CREDENTIALS_FILE}. Alternatively set LDP_VM_CREDENTIALS_FILE.
EOF
    exit 1
fi

# GitHub and ntfy are optional for public clones and notification-free runs.
GITHUB_TOKEN="${GITHUB_TOKEN:-}"
NTFY_AUTH_TOKEN="${NTFY_AUTH_TOKEN:-}"
: "${WANDB_API_KEY:?WANDB_API_KEY must be set in ${CREDENTIALS_FILE} or the environment}"

readonly MINIFORGE_VERSION="24.11.3-0"
readonly MINIFORGE_BASE_URL="https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}"

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

on_exit() {
    local exit_status=$?
    trap - EXIT
    set +e
    [[ -n "${GIT_ASKPASS_FILE:-}" ]] && rm -f "${GIT_ASKPASS_FILE}" || true
    if (( exit_status != 0 )); then
        [[ -n "${NTFY_AUTH_TOKEN}" ]] && curl --fail --silent --show-error --max-time 20 \
            -H "Authorization: Bearer ${NTFY_AUTH_TOKEN}" -H "Title: LDP setup failed" -H "Tags: warning" \
            --data-binary "LDP setup failed on $(hostname): exit_status=${exit_status}" \
            "${NTFY_SERVER%/}/${NTFY_TOPIC}" || true
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

log "Installing system packages"
log "Dataset root selected by free space: ${DATA_ROOT}"
as_root apt-get update
as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential ca-certificates cmake curl ffmpeg git git-lfs \
    libegl1 libgl1 libglib2.0-0 libglfw3 libglew-dev libosmesa6-dev \
    patchelf pkg-config rsync screen unzip wget
# Permit CUDA pinned-memory allocations for this VM user. The limits file
# applies after the next login; prlimit attempts to update this shell now.
MEMLOCK_USER="${SUDO_USER:-${USER}}"
printf '%s soft memlock unlimited\n%s hard memlock unlimited\n' "${MEMLOCK_USER}" "${MEMLOCK_USER}" \
    | as_root tee /etc/security/limits.d/99-ldp-memlock.conf >/dev/null
as_root prlimit --pid "$$" --memlock=unlimited:unlimited || true
ulimit -l unlimited || true
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
MINIFORGE_ARCH="x86_64"

if [[ "${RUN_SETUP}" == "true" ]]; then
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

else
    log "Skipping setup; reusing ${REPO_DIR}, ${CONDA_ENV_NAME}, and existing data"
fi

source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
cd "${REPO_DIR}"

log "Logging into Weights & Biases"
export WANDB_API_KEY WANDB_ENTITY
python -c 'import os, wandb; wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)'

log "Verifying the environment"
python -c 'import torch, wandb, diffusion_policy; print(f"Python setup OK; torch={torch.__version__}; CUDA available={torch.cuda.is_available()}; wandb={wandb.__version__}")'

AUTH_FILE="${REPO_DIR}/.vm_auth.env"
(
    umask 077
    {
        printf 'export WANDB_API_KEY=%q\n' "${WANDB_API_KEY}"
        printf 'export WANDB_ENTITY=%q\n' "${WANDB_ENTITY}"
        printf 'export NTFY_AUTH_TOKEN=%q\n' "${NTFY_AUTH_TOKEN}"
        printf 'export NTFY_SERVER=%q\n' "${NTFY_SERVER}"
        printf 'export NTFY_TOPIC=%q\n' "${NTFY_TOPIC}"
    } >"${AUTH_FILE}"
)
log "Environment setup complete. Run scripts/train_vm.sh to train."
