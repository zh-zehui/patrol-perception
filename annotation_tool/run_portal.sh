#!/usr/bin/env bash
set -euo pipefail

ANNOTATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${D1_AUTOLABEL_ENV_FILE:-${HOME}/.d1-autolabel-campus.env}"
REGISTRY_FILE="${D1_DATASET_REGISTRY:-${ANNOTATION_DIR}/config/datasets_registry.yaml}"
PORTAL_HOST="${D1_PORTAL_HOST:-0.0.0.0}"
PORTAL_PORT="${D1_PORTAL_PORT:-8765}"
PUBLIC_HOST="${D1_PUBLIC_HOST:-}"
ANNOTATION_PYTHON="${D1_ANNOTATION_PYTHON:-/home/sribd/miniforge3/envs/d1-autolabel/bin/python}"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

cd "${ANNOTATION_DIR}"
exec "${ANNOTATION_PYTHON}" -m d1_annotation.portal_cli \
  --registry "${REGISTRY_FILE}" \
  --host "${PORTAL_HOST}" \
  --port "${PORTAL_PORT}" \
  --public-host "${PUBLIC_HOST}"
