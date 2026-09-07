#!/usr/bin/env bash
# 登记 job 的 remap + append / remove 入口（见 datasets/DOWNLOAD_COMMANDS.md §6）。
# 用法：merge_job.sh list | append <job> [--keep A,B] [-y] [--dry-run] | remove <job> [--drop A,B] [-y] [--dry-run]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export DATA_ROOT="${DATA_ROOT:-${PATROL_DATA_ROOT:-/media/sribd/C02E01372E01284C/patrol_perception}}"
exec "${PYTHON:-python}" datasets/scripts/merge_job_cli.py "$@"
