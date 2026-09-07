from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml


@lru_cache(maxsize=1)
def _load_classes_cfg() -> dict:
    root = Path(__file__).resolve().parents[3]  # patrol_perception/
    path = root / "config" / "classes.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def map_yolo_name_to_detect_name(yolo_name: str) -> Optional[str]:
    """
    Map Ultralytics COCO class names to patrol_perception detect_names.

    W0 uses COCO-pretrained yolo26s.pt, so we map COCO -> our names:
    - exact match: e.g., "person" -> "person"
    - alias in classes.yaml.coco_alias: e.g., "bottle" -> "litter_bottle"
    - basic normalization: "traffic light" -> "traffic_light"
    """

    cfg = _load_classes_cfg()
    detect_names = set(cfg.get("detect_names") or [])
    coco_alias = cfg.get("coco_alias") or {}

    name = (yolo_name or "").strip()
    if not name:
        return None

    if name in detect_names:
        return name

    if name in coco_alias:
        return coco_alias[name]

    normalized = name.replace(" ", "_")
    if normalized in detect_names:
        return normalized

    return None

