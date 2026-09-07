from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from patrol_perception.types import Detection, XYXY


def iou_xyxy(a: XYXY, b: XYXY) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return float(inter_area / union)


@dataclass
class Track:
    track_id: int
    cls: str
    bbox: XYXY
    age: int = 0
    missed: int = 0


class IoUTracker:
    """
    Minimal IoU-based multi-object tracker.

    - Tracks are per-class (cls must match to match detections).
    - Greedy matching: highest IoU over threshold.
    - Drop tracks after missed > max_age.
    """

    def __init__(self, iou_thresh: float = 0.3, max_age: int = 30):
        self.iou_thresh = float(iou_thresh)
        self.max_age = int(max_age)
        self._next_id = 1
        self._tracks: Dict[int, Track] = {}

    def update(self, detections: List[Detection]) -> List[Detection]:
        # Step 1: mark all tracks as missed; then match detections to tracks.
        for t in self._tracks.values():
            t.missed += 1
            t.age += 1

        # Step 2: greedy matching per class
        for cls in sorted({d.cls for d in detections}):
            dets = [d for d in detections if d.cls == cls]
            tracks = [t for t in self._tracks.values() if t.cls == cls]
            if not dets:
                continue
            if not tracks:
                # First frame for this class: create tracks directly.
                for det in dets:
                    track_id = self._next_id
                    self._next_id += 1
                    self._tracks[track_id] = Track(
                        track_id=track_id, cls=cls, bbox=det.xyxy, age=0, missed=0
                    )
                    det.track_id = track_id
                continue

            # compute IoU matrix
            best_pairs: List[Tuple[float, int, int]] = []
            track_ids = [t.track_id for t in tracks]
            det_idxs = list(range(len(dets)))
            for di in det_idxs:
                for ti, track_id in enumerate(track_ids):
                    det_bbox = dets[di].xyxy
                    tr_bbox = self._tracks[track_id].bbox
                    best_pairs.append((iou_xyxy(det_bbox, tr_bbox), di, track_id))
            # sort descending IoU
            best_pairs.sort(key=lambda x: x[0], reverse=True)

            used_det = set()
            used_track = set()
            for score, di, track_id in best_pairs:
                if score < self.iou_thresh:
                    break
                if di in used_det or track_id in used_track:
                    continue
                # assign
                det = dets[di]
                det.track_id = track_id
                tr = self._tracks[track_id]
                tr.bbox = det.xyxy
                tr.missed = 0
                used_det.add(di)
                used_track.add(track_id)

            # Step 3: create new tracks for unmatched detections
            for di, det in enumerate(dets):
                if di in used_det:
                    continue
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = Track(
                    track_id=track_id, cls=cls, bbox=det.xyxy, age=0, missed=0
                )
                det.track_id = track_id

        # Step 4: prune dead tracks
        dead = [tid for tid, t in self._tracks.items() if t.missed > self.max_age]
        for tid in dead:
            del self._tracks[tid]

        return detections

