from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Set, Tuple

import numpy as np

from patrol_perception.types import Detection, Event


@dataclass
class NightLoiteringRuleConfig:
    # Night gate by mean luminance of the frame (simple heuristic for offline W0).
    night_mean_threshold: float = 60.0

    # Track dwell/motion gate
    history_len: int = 15
    min_dwell_frames: int = 10
    motion_max_px: float = 25.0


class NightLoiteringRule:
    def __init__(self, cfg: NightLoiteringRuleConfig):
        self.cfg = cfg
        # track_id -> deque of centers
        self._hist: Dict[int, Deque[Tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=self.cfg.history_len)
        )

    @staticmethod
    def _frame_mean_luma(image_bgr: np.ndarray) -> float:
        # Use Y channel from BGR -> HSV-like luminance
        y = 0.114 * image_bgr[:, :, 0] + 0.587 * image_bgr[:, :, 1] + 0.299 * image_bgr[:, :, 2]
        return float(np.mean(y))

    def update(
        self,
        image_bgr: np.ndarray,
        detections: List[Detection],
        active_event_ids: Set[str],
    ) -> List[Event]:
        if "night_loitering" not in active_event_ids:
            return []

        mean_luma = self._frame_mean_luma(image_bgr)
        if mean_luma >= self.cfg.night_mean_threshold:
            return []

        persons = [d for d in detections if d.cls == "person" and d.track_id is not None]
        if not persons:
            return []

        # update history
        for d in persons:
            tid = int(d.track_id)
            cx, cy = d.center()
            self._hist[tid].append((cx, cy))

        # decide events
        events: List[Event] = []
        for tid, centers in list(self._hist.items()):
            if len(centers) < self.cfg.min_dwell_frames:
                continue
            xs = [c[0] for c in centers]
            ys = [c[1] for c in centers]
            motion = (max(xs) - min(xs)) + (max(ys) - min(ys))
            if motion <= self.cfg.motion_max_px:
                events.append(
                    Event(
                        type="night_loitering",
                        layer="s20",
                        score=1.0,
                        maturity="",
                        track_ids=[tid],
                        evidence={
                            "mean_luma": mean_luma,
                            "num_frames": len(centers),
                            "motion_px": motion,
                        },
                    )
                )
        return events

