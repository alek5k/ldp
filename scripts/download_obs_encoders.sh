#!/usr/bin/env bash
# Download the shared PTP observation-encoder bundle from the project README.
set -Eeuo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_file="${1:-}"
if [[ -n "${target_file}" && -f "${repo_dir}/obs_encoders/${target_file}" ]]; then
    exit 0
fi

archive="${repo_dir}/obs_encoders.zip"
html_file="$(mktemp)"
cookie_file="$(mktemp)"
cleanup() { rm -f "${html_file}" "${cookie_file}" "${archive}"; }
trap cleanup EXIT

echo "Downloading shared observation encoders..."
wget --quiet --save-cookies "${cookie_file}" --keep-session-cookies \
    "https://drive.google.com/uc?export=download&id=1tSYyWg3HZbTtEhzpAXQpl28DSrWsXc7J" \
    -O "${html_file}"
uuid="$(grep -oP 'name="uuid" value="\K[^"]+' "${html_file}" || true)"
confirm="$(grep -oP 'name="confirm" value="\K[^"]+' "${html_file}" || true)"
[[ -n "${uuid}" && -n "${confirm}" ]] || { echo "Google Drive confirmation changed; see README.md." >&2; exit 1; }
wget --show-progress --load-cookies "${cookie_file}" \
    "https://drive.usercontent.google.com/download?id=1tSYyWg3HZbTtEhzpAXQpl28DSrWsXc7J&export=download&confirm=${confirm}&uuid=${uuid}" \
    -O "${archive}"
unzip -q -o "${archive}" -d "${repo_dir}"
[[ -z "${target_file}" || -f "${repo_dir}/obs_encoders/${target_file}" ]] || { echo "Encoder not found after extraction: ${target_file}" >&2; exit 1; }
