from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from patrol_perception.types import Detection, Event, FrameResult


def detection_to_dict(d: Detection) -> Dict[str, Any]:
    return {
        "cls": d.cls,
        "conf": d.conf,
        "xyxy": list(d.xyxy),
        "track_id": d.track_id,
    }


def event_to_dict(e: Event) -> Dict[str, Any]:
    return {
        "type": e.type,
        "layer": e.layer,
        "score": e.score,
        "maturity": e.maturity,
        "track_ids": list(e.track_ids),
        "evidence": e.evidence,
    }


def frame_result_to_dict(fr: FrameResult) -> Dict[str, Any]:
    return {
        "timestamp": fr.timestamp,
        "frame_id": fr.frame_id,
        "detections": [detection_to_dict(d) for d in fr.detections],
        "events": [event_to_dict(e) for e in fr.events],
        "meta": fr.meta,
    }


def save_frame_result_json(fr: FrameResult, path: Path) -> None:
    path.write_text(
        json.dumps(frame_result_to_dict(fr), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

