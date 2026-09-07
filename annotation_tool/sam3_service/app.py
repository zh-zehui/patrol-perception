from __future__ import annotations

import io
import json
import os
import threading
from typing import Any

from fastapi import FastAPI, Form, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

app = FastAPI(title="D1 SAM 3 Service", version="0.1.0")

_runtime_lock = threading.Lock()
_inference_lock = threading.Lock()
_runtime: tuple[Any, Any, Any] | None = None
_load_error: str | None = None


def _load_runtime() -> tuple[Any, Any, Any]:
    global _runtime, _load_error
    if _runtime is not None:
        return _runtime
    with _runtime_lock:
        if _runtime is not None:
            return _runtime
        try:
            import torch
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available inside the SAM 3 service")
            checkpoint = os.getenv("SAM3_CHECKPOINT") or None
            model = build_sam3_image_model(
                device="cuda",
                checkpoint_path=checkpoint,
                load_from_HF=checkpoint is None,
                enable_segmentation=True,
                compile=os.getenv("SAM3_COMPILE", "false").lower() in {"1", "true", "yes"},
            )
            processor = Sam3Processor(
                model,
                device="cuda",
                confidence_threshold=float(os.getenv("SAM3_CONFIDENCE_THRESHOLD", "0.5")),
            )
            _runtime = torch, model, processor
            _load_error = None
            return _runtime
        except Exception as exc:
            _load_error = f"{type(exc).__name__}: {exc}"
            raise


def _iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    x1 = max(left["x_min"], right["x_min"])
    y1 = max(left["y_min"], right["y_min"])
    x2 = min(left["x_max"], right["x_max"])
    y2 = min(left["y_max"], right["y_max"])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0
    left_area = (left["x_max"] - left["x_min"]) * (left["y_max"] - left["y_min"])
    right_area = (right["x_max"] - right["x_min"]) * (right["y_max"] - right["y_min"])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _smaller_box_containment(left: dict[str, Any], right: dict[str, Any]) -> float:
    """Return how much of the smaller box is covered by the other box."""
    x1 = max(left["x_min"], right["x_min"])
    y1 = max(left["y_min"], right["y_min"])
    x2 = min(left["x_max"], right["x_max"])
    y2 = min(left["y_max"], right["y_max"])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0
    left_area = (left["x_max"] - left["x_min"]) * (left["y_max"] - left["y_min"])
    right_area = (right["x_max"] - right["x_min"]) * (right["y_max"] - right["y_min"])
    smaller_area = min(left_area, right_area)
    return intersection / smaller_area if smaller_area > 0 else 0.0


def _class_nms(
    detections: list[dict[str, Any]], threshold: float, containment_threshold: float = 0.9
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for detection in sorted(detections, key=lambda item: item["confidence"], reverse=True):
        if any(
            other["class_id"] == detection["class_id"]
            and (
                _iou(other, detection) >= threshold
                or _smaller_box_containment(other, detection) >= containment_threshold
            )
            for other in kept
        ):
            continue
        kept.append(detection)
    return kept


def _resolve_bicycle_ebike_conflicts(
    detections: list[dict[str, Any]], threshold: float = 0.85
) -> list[dict[str, Any]]:
    """Resolve class 1/6 duplicates that describe the same physical vehicle.

    The high IoU threshold preserves distinct bicycles and ebikes that merely
    overlap because they are riding or parked next to each other.
    """
    kept: list[dict[str, Any]] = []
    for detection in sorted(detections, key=lambda item: item["confidence"], reverse=True):
        conflicts = detection["class_id"] in {1, 6} and any(
            other["class_id"] in {1, 6}
            and other["class_id"] != detection["class_id"]
            and _iou(other, detection) >= threshold
            for other in kept
        )
        if not conflicts:
            kept.append(detection)
    return kept


def _parse_classes(classes_json: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(classes_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="classes_json 不是有效 JSON") from exc
    if not isinstance(value, list) or not value:
        raise HTTPException(status_code=422, detail="classes_json 必须是非空类别数组")
    classes: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict) or not isinstance(item.get("id"), int):
            raise HTTPException(status_code=422, detail="每个类别必须包含整数 id")
        prompts = item.get("prompts") or [item.get("name")]
        prompts = [str(prompt).strip() for prompt in prompts if str(prompt).strip()]
        if not prompts:
            continue
        classes.append({"id": item["id"], "name": str(item.get("name", item["id"])),
                        "prompts": list(dict.fromkeys(prompts))[:4]})
    if not classes:
        raise HTTPException(status_code=422, detail="没有可用于 SAM 3 的文字提示")
    return classes


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": _load_error is None,
        "model": os.getenv("SAM3_MODEL", "sam3"),
        "loaded": _runtime is not None,
        "load_error": _load_error,
    }


@app.post("/v1/load")
def load_model() -> dict[str, Any]:
    try:
        torch, _model, _processor = _load_runtime()
        return {"ok": True, "model": os.getenv("SAM3_MODEL", "sam3"),
                "device": torch.cuda.get_device_name(0)}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"SAM 3 加载失败：{type(exc).__name__}: {exc}") from exc


@app.post("/v1/predict")
def predict(
    image: UploadFile,
    classes_json: str = Form(...),
    confidence_threshold: float = Form(0.5, ge=0.0, le=1.0),
    nms_iou_threshold: float = Form(0.7, ge=0.0, le=1.0),
) -> dict[str, Any]:
    max_upload = int(os.getenv("SAM3_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
    raw = image.file.read(max_upload + 1)
    if len(raw) > max_upload:
        raise HTTPException(status_code=413, detail="图片文件过大")
    try:
        source = Image.open(io.BytesIO(raw))
        prepared = ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=422, detail="无法读取上传图片") from exc
    classes = _parse_classes(classes_json)
    try:
        torch, _model, processor = _load_runtime()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"SAM 3 加载失败：{type(exc).__name__}: {exc}") from exc

    detections: list[dict[str, Any]] = []
    # SAM 3's official image notebook runs the full forward pass under CUDA
    # BF16 autocast. Without this context, its fused MLP can produce BF16
    # activations for FP32 linear weights and fail with a dtype mismatch.
    with _inference_lock, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        processor.set_confidence_threshold(confidence_threshold)
        state = processor.set_image(prepared)
        for class_entry in classes:
            for prompt in class_entry["prompts"]:
                processor.reset_all_prompts(state)
                output = processor.set_text_prompt(prompt=prompt, state=state)
                boxes = output["boxes"].detach().float().cpu().tolist()
                scores = output["scores"].detach().float().cpu().tolist()
                for box, score in zip(boxes, scores):
                    x_min, y_min, x_max, y_max = (float(value) for value in box)
                    if x_max <= x_min or y_max <= y_min:
                        continue
                    detections.append({
                        "class_id": class_entry["id"],
                        "class_name": class_entry["name"],
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max,
                        "confidence": float(score),
                        "prompt": prompt,
                    })
        processor.reset_all_prompts(state)

    detections = _class_nms(detections, nms_iou_threshold)
    detections = _resolve_bicycle_ebike_conflicts(detections)
    max_detections = int(os.getenv("SAM3_MAX_DETECTIONS", "1000"))
    detections = sorted(detections, key=lambda item: item["confidence"], reverse=True)[:max_detections]
    return {
        "model": os.getenv("SAM3_MODEL", "sam3"),
        "width": prepared.width,
        "height": prepared.height,
        "detections": detections,
    }
