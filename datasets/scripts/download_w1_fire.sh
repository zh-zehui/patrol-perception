#!/usr/bin/env bash
# W1 烟火：下载 HF 集（若尚未）→ remap → 与 COCO train2017(+val2017) 合并为 detect_w1_fire
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

DATA_ROOT="${PATROL_DATA_ROOT:-${1:-/media/sribd/C02E01372E01284C/patrol_perception}}"
PY="${PYTHON:-python}"
# 默认拉 train2017；可 COCO_VARIANT=val2017 退回仅 val 回放
COCO_VARIANT="${COCO_VARIANT:-train2017}"

echo "[w1] data_root=$DATA_ROOT coco_variant=$COCO_VARIANT"

# 1) 下载烟火 + COCO（train2017 会落到 coco_train2017，不覆盖 val）
PYTHONPATH=src "$PY" datasets/scripts/download_from_sources.py \
  --classes smoke,flame,person,bicycle,car,truck \
  --data-root "$DATA_ROOT" \
  --coco-variant "$COCO_VARIANT"

# 若用 train2017，仍确保 val2017 存在作验证集
if [[ "$COCO_VARIANT" == "train2017" ]]; then
  PYTHONPATH=src "$PY" datasets/scripts/download_from_sources.py \
    --classes person,bicycle,car,truck \
    --data-root "$DATA_ROOT" \
    --coco-variant val2017
fi

FIRE_RAW="$DATA_ROOT/datasets/downloads/fog_fire_detection/dataset"
FIRE_YOLO="$DATA_ROOT/datasets/downloads/fog_fire_yolo"
COCO_VAL="$DATA_ROOT/datasets/downloads/coco_val2017"
COCO_TRAIN="$DATA_ROOT/datasets/downloads/coco_train2017"
OUT="$DATA_ROOT/datasets/detect_w1_fire"

if [[ ! -d "$FIRE_RAW" ]]; then
  echo "[w1] missing $FIRE_RAW — check HF download / .hf_token" >&2
  exit 1
fi

# 2) 映射 Fire/Smoke → flame/smoke
"$PY" datasets/scripts/remap_yolo_set.py \
  --src "$FIRE_RAW" \
  --out "$FIRE_YOLO" \
  --map config/class_maps/fire_smoke.yaml

# 3) 合并 → detect_w1_fire
TMP_CFG="$(mktemp)"
trap 'rm -f "$TMP_CFG"' EXIT

if [[ "$COCO_VARIANT" == "train2017" ]]; then
  if [[ ! -f "$COCO_TRAIN/data.yaml" ]]; then
    echo "[w1] missing coco train2017 at $COCO_TRAIN" >&2
    exit 1
  fi
  if [[ ! -f "$COCO_VAL/data.yaml" ]]; then
    echo "[w1] missing coco val2017 at $COCO_VAL" >&2
    exit 1
  fi
  cat >"$TMP_CFG" <<EOF
version: 1
names:
  - person
  - bicycle
  - car
  - truck
  - smoke
  - flame
sets:
  - root: $FIRE_YOLO
    prefix: fire
    keep: [flame, smoke]
    split_ratios: [0.7, 0.2, 0.1]
    split_seed: 0
  - root: $COCO_TRAIN
    extra_roots:
      - $COCO_VAL
    prefix: coco
    keep: [person, bicycle, car, truck]
    split_ratios: [0.7, 0.2, 0.1]
    split_seed: 0
EOF
else
  if [[ ! -f "$COCO_VAL/data.yaml" ]]; then
    echo "[w1] missing coco val2017 at $COCO_VAL" >&2
    exit 1
  fi
  cat >"$TMP_CFG" <<EOF
version: 1
names:
  - person
  - bicycle
  - car
  - truck
  - smoke
  - flame
sets:
  - root: $FIRE_YOLO
    prefix: fire
    keep: [flame, smoke]
    split_ratios: [0.7, 0.2, 0.1]
    split_seed: 0
  - root: $COCO_VAL
    prefix: coco
    keep: [person, bicycle, car, truck]
    split_ratios: [0.7, 0.2, 0.1]
    split_seed: 0
EOF
fi

"$PY" datasets/scripts/merge_yolo_sets.py --config "$TMP_CFG" --out "$OUT"

echo "[w1] done → $OUT/data.yaml"
echo "[w1] train example:"
echo "  yolo detect train model=data/weights/yolo26s.pt data=$OUT/data.yaml imgsz=640 epochs=50"
