from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Set

import cv2
import numpy as np

from patrol_perception.catalog import iter_active_events, load_catalog
from patrol_perception.models.detect import YoloDetectConfig, YoloDetector
from patrol_perception.pipeline.json_io import save_frame_result_json
from patrol_perception.rules.coco_discover import coco_discover_events
from patrol_perception.rules.crowd import crowd_events
from patrol_perception.rules.night_loitering import NightLoiteringRule, NightLoiteringRuleConfig
from patrol_perception.track.simple_tracker import IoUTracker
from patrol_perception.types import FrameResult


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(source: Path) -> List[Path]:
    if source.is_file():
        return [source]
    items = [p for p in sorted(source.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    if not items:
        raise FileNotFoundError(f"no images under {source}")
    return items


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="image file or directory")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--weights", default="data/weights/yolo26s.pt", help="YOLO26 weights path")
    p.add_argument("--device", default="cuda:0", help="cuda:0 or cpu")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.3, help="tracker iou threshold")
    p.add_argument("--max-age", type=int, default=30, help="tracker max missed frames")
    p.add_argument("--max-frames", type=int, default=0, help="0 means all")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(__file__).resolve().parents[3]
    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = root / weights_path

    source = Path(args.source)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cat = load_catalog(root / "config" / "events_catalog.yaml")
    active_event_ids: Set[str] = {e["id"] for e in iter_active_events(cat)}

    detector = YoloDetector(
        YoloDetectConfig(
            weights=str(weights_path),
            device=args.device,
            conf=args.conf,
            imgsz=args.imgsz,
        )
    )
    tracker = IoUTracker(iou_thresh=args.iou, max_age=args.max_age)
    night_rule = NightLoiteringRule(NightLoiteringRuleConfig())

    image_paths = list_images(source)
    if args.max_frames and args.max_frames > 0:
        image_paths = image_paths[: args.max_frames]

    all_events: List[Dict[str, Any]] = []
    for frame_id, img_path in enumerate(image_paths):
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            continue
        timestamp = time.time()

        detections = detector.predict(image_bgr)
        detections = tracker.update(detections)

        events = []
        events.extend(coco_discover_events(detections, active_event_ids=active_event_ids))
        events.extend(crowd_events(detections, active_event_ids=active_event_ids))
        events.extend(night_rule.update(image_bgr=image_bgr, detections=detections, active_event_ids=active_event_ids))

        fr = FrameResult(
            timestamp=timestamp,
            frame_id=frame_id,
            detections=detections,
            events=events,
            meta={
                "source": str(img_path),
                "mean_luma_gate": None,
            },
        )
        save_frame_result_json(fr, out_dir / f"frame_{frame_id:06d}.json")

        for e in events:
            all_events.append(
                {
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "event": {
                        "type": e.type,
                        "layer": e.layer,
                        "score": e.score,
                        "maturity": e.maturity,
                        "track_ids": e.track_ids,
                        "evidence": e.evidence,
                    },
                }
            )

    (out_dir / "events.json").write_text(json.dumps(all_events, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[offline_run] done. frames={len(image_paths)} out={out_dir}")


if __name__ == "__main__":
    main()

