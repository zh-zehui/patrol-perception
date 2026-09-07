from __future__ import annotations

from typing import Dict, List, Set

from patrol_perception.types import Detection, Event


DETECTION_TO_EVENT = {
    "bicycle": "bicycle",
    "car": "car",
    "truck": "truck",
    "pet": "pet",
    "litter_bottle": "litter_bottle",
}


def coco_discover_events(
    detections: List[Detection],
    active_event_ids: Set[str],
    conf_min: float = 0.5,
) -> List[Event]:
    """
    W0: pure "discover" events for COCO-origin classes.
    """

    events: List[Event] = []
    by_type: Dict[str, List[Detection]] = {}
    for d in detections:
        if d.conf < conf_min:
            continue
        ev_type = DETECTION_TO_EVENT.get(d.cls)
        if not ev_type or ev_type not in active_event_ids:
            continue
        by_type.setdefault(ev_type, []).append(d)

    for ev_type, dets in by_type.items():
        track_ids = sorted({d.track_id for d in dets if d.track_id is not None})
        score = max(d.conf for d in dets) if dets else 0.0
        evidence = {
            "num": len(dets),
            "tracks": track_ids,
            "boxes": [{"xyxy": list(d.xyxy), "conf": d.conf} for d in dets],
        }
        events.append(
            Event(
                type=ev_type,
                layer="s20",
                score=float(score),
                track_ids=[int(t) for t in track_ids],
                evidence=evidence,
            )
        )
    return events

