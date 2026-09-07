from __future__ import annotations

from typing import List, Set

from patrol_perception.types import Detection, Event


def crowd_events(
    detections: List[Detection],
    active_event_ids: Set[str],
    conf_min: float = 0.5,
    crowd_person_n: int = 5,
) -> List[Event]:
    """
    crowd: simple per-frame person counting.

    W0 scope: use person detections only, no extra density grid.
    """

    if "crowd" not in active_event_ids:
        return []

    persons = [d for d in detections if d.cls == "person" and d.conf >= conf_min]
    track_ids = sorted({d.track_id for d in persons if d.track_id is not None})
    n = len(track_ids) if track_ids else len(persons)

    if n < crowd_person_n:
        return []

    evidence = {
        "num_person": n,
        "tracks": [int(t) for t in track_ids],
        "boxes": [{"xyxy": list(d.xyxy), "conf": d.conf, "track_id": d.track_id} for d in persons],
    }
    return [
        Event(
            type="crowd",
            layer="s20",
            score=1.0,
            maturity="",
            track_ids=[int(t) for t in track_ids],
            evidence=evidence,
        )
    ]

