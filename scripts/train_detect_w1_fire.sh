#!/usr/bin/env bash
# W1 烟火检测微调：yolo26l → detect_w1.pt，训后可选 eval（默认开启）。
#
# 前置：detect_w1_fire/data.yaml 已由 merge（见 datasets/scripts/download_w1_fire.sh
# 或 config/merge_w1_fire.yaml）生成；训后评测见 scripts/eval_detect_run.py。
#
# 用法（patrol_perception/ 下）:
#   ./scripts/train_detect_w1_fire.sh
#   EVAL_AFTER=0 ./scripts/train_detect_w1_fire.sh   # 只训练，跳过评测
#
# 环境变量：PATROL_DATA_ROOT, DATA_YAML, WEIGHTS, RUN_NAME, EPOCHS, BATCH,
#           EVAL_AFTER, EVAL_BASELINE（覆盖 eval.yaml baseline.default）,
#           EVAL_NO_BASELINE（=1 时评测不加遗忘基线）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA_ROOT="${PATROL_DATA_ROOT:-/media/sribd/C02E01372E01284C/patrol_perception}"
DATA_YAML="${DATA_YAML:-$DATA_ROOT/datasets/detect_w1_fire/data.yaml}"
WEIGHTS="${WEIGHTS:-$ROOT/data/weights/yolo26l.pt}"
RUN_NAME="${RUN_NAME:-w1_fire}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-16}"
EVAL_AFTER="${EVAL_AFTER:-1}"
# 训后 eval 的 wave 键，须与 config/eval.yaml focus_classes_by_wave 一致
# 联合 15 类重训示例：WAVE=w2_joint RUN_NAME=w2_joint ./scripts/train_detect_w1_fire.sh
WAVE="${WAVE:-w1_fire}"

CUDA_SWITCH="/home/sribd/D1_EDU/Pytorch/switch_cuda_mode.sh"
if [[ ! -f "$CUDA_SWITCH" ]]; then
  echo "[w1-train] missing $CUDA_SWITCH" >&2
  exit 1
fi
# shellcheck source=/home/sribd/D1_EDU/Pytorch/switch_cuda_mode.sh
source "$CUDA_SWITCH" 1

if [[ ! -f "$DATA_YAML" ]]; then
  echo "[w1-train] missing data.yaml: $DATA_YAML" >&2
  echo "[w1-train] run datasets/scripts/download_w1_fire.sh or merge_yolo_sets.py first" >&2
  exit 1
fi
if [[ ! -f "$WEIGHTS" ]]; then
  echo "[w1-train] missing weights: $WEIGHTS" >&2
  exit 1
fi

echo "[w1-train] data=$DATA_YAML weights=$WEIGHTS epochs=$EPOCHS batch=$BATCH eval_after=$EVAL_AFTER wave=$WAVE"

yolo detect train \
  model="$WEIGHTS" \
  data="$DATA_YAML" \
  imgsz=640 epochs="$EPOCHS" batch="$BATCH" \
  project="$ROOT/data/runs/detect" name="$RUN_NAME"

BEST="$ROOT/data/runs/detect/$RUN_NAME/weights/best.pt"
if [[ ! -f "$BEST" ]]; then
  echo "[w1-train] missing best.pt: $BEST" >&2
  exit 1
fi

mkdir -p "$ROOT/data/weights"
cp -f "$BEST" "$ROOT/data/weights/detect_w1.pt"
echo "[w1-train] copied → data/weights/detect_w1.pt"

if [[ "$EVAL_AFTER" == "1" ]]; then
  echo "[w1-train] running post-train eval (set EVAL_AFTER=0 to skip)"
  EVAL_ARGS=(
    --weights "$BEST"
    --data "$DATA_YAML"
    --wave "$WAVE"
    --config config/eval.yaml
  )
  if [[ "${EVAL_NO_BASELINE:-0}" == "1" ]]; then
    EVAL_ARGS+=(--no-baseline)
  elif [[ -n "${EVAL_BASELINE:-}" ]]; then
    EVAL_ARGS+=(--baseline "$EVAL_BASELINE")
  fi
  PYTHONPATH=src python scripts/eval_detect_run.py "${EVAL_ARGS[@]}"
else
  echo "[w1-train] EVAL_AFTER=0 — skipped post-train eval"
fi
