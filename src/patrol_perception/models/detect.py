from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from patrol_perception.types import Detection, XYXY
from patrol_perception.utils.mapping import map_yolo_name_to_detect_name


@dataclass
class YoloDetectConfig:
    weights: str
    device: str = "cuda:0"
    conf: float = 0.25
    imgsz: int = 640


class YoloDetector:
    """
    Thin Ultralytics YOLO wrapper.

    For offline W0 we rely on COCO-pretrained yolo26s.pt output.
    We map COCO class names to our detect_names using:
    - exact match
    - classes.yaml coco_alias (only bottle->litter_bottle)
    """

    def __init__(self, cfg: YoloDetectConfig):
        from ultralytics import YOLO

        self.cfg = cfg
        self.model = YOLO(cfg.weights)

        # ultralytics model.names is a dict: {id: name}
        self.id_to_name = dict(self.model.names)

    def predict(self, image_bgr: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            image_bgr,
            device=self.cfg.device,
            conf=self.cfg.conf,
            imgsz=self.cfg.imgsz,
            verbose=False,
        )
        r = results[0]
        dets: List[Detection] = []
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            coco_name = self.id_to_name.get(cls_id, str(cls_id))
            mapped = map_yolo_name_to_detect_name(coco_name)
            if mapped is None:
                continue
            dets.append(Detection(cls=mapped, conf=conf, xyxy=(x1, y1, x2, y2)))
        return dets

