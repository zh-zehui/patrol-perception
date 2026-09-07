from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


XYXY = Tuple[float, float, float, float]


@dataclass
class Detection:
    """
    Unified detection record used by tracker and event rules.
    xyxy: (x1, y1, x2, y2) in original image pixel coordinates.
    """

    cls: str
    conf: float
    xyxy: XYXY
    track_id: Optional[int] = None

    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class Event:
    """
    Event output contract (minimal for offline W0).
    type: event id (English id in events_catalog.yaml)
    """

    type: str
    layer: str
    score: float = 0.0
    maturity: str = ""
    track_ids: List[int] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameResult:
    timestamp: float
    frame_id: int
    detections: List[Detection] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

