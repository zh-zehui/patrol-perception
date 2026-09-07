#!/usr/bin/env bash
set -euo pipefail

SAM3_SERVICE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /home/sribd/miniforge3/etc/profile.d/conda.sh
conda activate d1-sam3

export HF_HOME="/home/sribd/.cache/huggingface"
export SAM3_MODEL="${SAM3_MODEL:-sam3}"
export SAM3_CONFIDENCE_THRESHOLD="${SAM3_CONFIDENCE_THRESHOLD:-0.5}"
export SAM3_CHECKPOINT="${SAM3_CHECKPOINT:-/home/sribd/models/sam3/sam3.pt}"

cd "${SAM3_SERVICE_DIR}"
exec python -m uvicorn app:app --host 127.0.0.1 --port 8783 --workers 1
