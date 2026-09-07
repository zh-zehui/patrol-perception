from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol

import httpx
from PIL import Image, ImageOps
from pydantic import BaseModel, Field, ValidationError


class AutoLabelError(RuntimeError):
    """Safe, user-actionable auto-label error without credentials or image data."""


class Detection(BaseModel):
    class_id: int = Field(ge=0)
    x_min: float = Field(ge=0, le=1000)
    y_min: float = Field(ge=0, le=1000)
    x_max: float = Field(ge=0, le=1000)
    y_max: float = Field(ge=0, le=1000)
    confidence: float = Field(default=0.5, ge=0, le=1)
    evidence: str = ""


class DetectionResult(BaseModel):
    detections: list[Detection]


class AutoLabelProvider(Protocol):
    name: str
    model: str

    def predict(self, image: Path, classes: list[dict[str, Any]]) -> list[dict[str, Any]]: ...


PROMPT = """You are producing candidate object-detection annotations for human review.
Inspect the image carefully and detect only objects belonging to the exact class list below.
Coordinates use a 0..1000 image coordinate system with origin at top-left.
Return a tight visible-object rectangle: x_min < x_max and y_min < y_max.
Do not invent objects or classes. Include every clearly visible instance separately.
Use confidence conservatively; uncertain or unlocalizable objects must be omitted.
Project-specific rules:
- class 12 cigarette_or_smoke means the physical cigarette, not a person or generic smoke.
- class 13 pet means cats and dogs; each animal gets its own box.
- class 4 smoke means obvious non-cigarette environmental smoke.
- litter_* requires an apparently discarded object, not an item normally held, stored, or displayed.
- class 1 bicycle requires bicycle geometry: visible pedals/crank and a chain (normally a
  diamond-frame pedal cycle). Do not use it for a step-through electric scooter or moped.
- class 6 ebike means the Chinese-style seated electric scooter or electric moped commonly
  used for commuting and deliveries. It has an overall motor-scooter appearance, often with
  a step-through body, footboard, battery compartment, small wheels, long seat, rear rack or
  storage box. These are supporting clues; no single component must be visible. It does not
  mean a Western pedal-assist bicycle. Use class 1 only when pedals, crank, chain, and a
  traditional bicycle frame are clearly visible. If the scooter is partly occluded, small,
  or some details are hidden, still use class 6 when its overall appearance matches a seated
  electric scooter/moped; do not omit it solely because individual details are hidden.
- for occluded objects, bound the visible region; do not guess the hidden extent.
Allowed classes:
{class_lines}
"""

def detection_schema_for(classes: list[dict[str, Any]]) -> dict[str, Any]:
    max_id = max((int(c["id"]) for c in classes), default=0)
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "detections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "class_id": {"type": "integer", "minimum": 0, "maximum": max_id},
                        "x_min": {"type": "number", "minimum": 0, "maximum": 1000},
                        "y_min": {"type": "number", "minimum": 0, "maximum": 1000},
                        "x_max": {"type": "number", "minimum": 0, "maximum": 1000},
                        "y_max": {"type": "number", "minimum": 0, "maximum": 1000},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "evidence": {"type": "string"},
                    },
                    "required": [
                        "class_id", "x_min", "y_min", "x_max", "y_max", "confidence", "evidence",
                    ],
                },
            }
        },
        "required": ["detections"],
    }


# Backward-compatible default; prefer detection_schema_for(classes).
DETECTION_SCHEMA: dict[str, Any] = detection_schema_for([{"id": i} for i in range(14)])


def prepare_image_data_url(image: Path, max_edge: int = 2048, quality: int = 90) -> str:
    """Normalize orientation and bound upload size without changing normalized coordinates."""
    with Image.open(image) as source:
        prepared = ImageOps.exif_transpose(source).convert("RGB")
        prepared.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        output = BytesIO()
        prepared.save(output, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _response_text(data: dict[str, Any]) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    for output in data.get("output", []):
        if not isinstance(output, dict):
            continue
        for content in output.get("content", []):
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                return content["text"]
    # A few OpenAI-compatible gateways wrap output as Chat Completions.
    choices = data.get("choices", [])
    if choices and isinstance(choices[0], dict):
        content = choices[0].get("message", {}).get("content")
        if isinstance(content, str):
            return content
    raise AutoLabelError("中转响应中没有可解析的 output_text")


def _json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
        # A few gateways double-encode structured output as a JSON string.
        if isinstance(value, str):
            value = json.loads(value)
    except json.JSONDecodeError as exc:
        raise AutoLabelError(f"模型没有返回有效 JSON（第 {exc.lineno} 行第 {exc.colno} 列）") from exc
    if isinstance(value, list):
        return {"detections": value}
    if not isinstance(value, dict):
        raise AutoLabelError("模型输出必须是 JSON 对象或检测框数组")
    if "detections" in value:
        return value
    # Some OpenAI-compatible relays/models ignore the requested root name.
    aliases = ("objects", "boxes", "annotations", "results", "items")
    for alias in aliases:
        if alias in value and isinstance(value[alias], list):
            return {"detections": value[alias]}
    keys = ", ".join(str(key) for key in list(value)[:8]) or "空对象"
    if any(key in value for key in ("refusal", "error", "message")):
        raise AutoLabelError(f"模型没有返回检测结果，响应字段为：{keys}")
    raise AutoLabelError(f"模型输出缺少 detections，实际顶层字段为：{keys}")


def _class_id(value: Any, classes: list[dict[str, Any]]) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        for entry in classes:
            if normalized == str(entry["name"]).lower():
                return int(entry["id"])
        aliases = {
            "cat": "pet", "dog": "pet", "bike": "bicycle",
            "electric bike": "ebike", "cigarette": "cigarette_or_smoke",
            "fire": "flame", "bottle": "litter_bottle",
            "plastic bag": "litter_plastic_bag", "paper": "litter_paper",
            "cardboard": "litter_cardboard", "fishing rod": "fishing_rod",
        }
        target = aliases.get(normalized)
        if target:
            return next((int(e["id"]) for e in classes if e["name"] == target), None)
    return None


def _number(value: Any) -> float | None:
    try:
        if isinstance(value, bool) or value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coordinates(item: dict[str, Any]) -> tuple[float, float, float, float] | None:
    # Accept xyxy under common names.
    for key in ("bbox", "box", "box_2d", "coordinates"):
        value = item.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 4:
            numbers = [_number(x) for x in value[:4]]
            if all(x is not None for x in numbers):
                return tuple(numbers)  # type: ignore[return-value]
        if isinstance(value, dict):
            nested = _coordinates(value)
            if nested:
                return nested
    for keys in (("x_min", "y_min", "x_max", "y_max"),
                 ("xmin", "ymin", "xmax", "ymax"),
                 ("left", "top", "right", "bottom"),
                 ("x1", "y1", "x2", "y2")):
        numbers = [_number(item.get(key)) for key in keys]
        if all(x is not None for x in numbers):
            return tuple(numbers)  # type: ignore[return-value]
    # Center/size format, accepted in either 0..1 or 0..1000 coordinates.
    for keys in (("x", "y", "width", "height"),
                 ("x_center", "y_center", "width", "height"),
                 ("cx", "cy", "w", "h")):
        center = [_number(item.get(key)) for key in keys]
        if all(x is not None for x in center):
            x, y, width, height = center
            return (x - width / 2, y - height / 2, x + width / 2, y + height / 2)
    return None


def _normalize_detections(value: dict[str, Any], classes: list[dict[str, Any]]) -> dict[str, Any]:
    raw = value.get("detections")
    if not isinstance(raw, list):
        return value
    normalized = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            normalized.append(item)
            continue
        copy = dict(item)
        class_id = _class_id(copy.get("class_id", copy.get("class", copy.get("label", copy.get("class_name")))), classes)
        if class_id is not None:
            copy["class_id"] = class_id
        coords = _coordinates(copy)
        if coords:
            # Accept normalized 0..1 coordinates as well as requested 0..1000.
            scale = 1000.0 if max(abs(value) for value in coords) <= 1.0 else 1.0
            copy["x_min"], copy["y_min"], copy["x_max"], copy["y_max"] = (
                value * scale for value in coords
            )
        else:
            keys = ", ".join(str(key) for key in list(copy)[:12]) or "空对象"
            raise AutoLabelError(f"第 {index + 1} 个检测框无法识别坐标，实际字段为：{keys}")
        if "confidence" not in copy:
            score = copy.get("score", copy.get("probability", copy.get("confidence_score")))
            copy["confidence"] = _number(score) if _number(score) is not None else 0.5
        if "evidence" not in copy:
            copy["evidence"] = str(copy.get("reason", copy.get("description", "")))
        normalized.append(copy)
    return {"detections": normalized}


def _validation_message(exc: ValidationError) -> str:
    details = []
    for error in exc.errors(include_url=False)[:3]:
        location = ".".join(str(part) for part in error.get("loc", ())) or "result"
        details.append(f"{location}: {error.get('msg', 'invalid value')}")
    return "；".join(details)


class OpenAIVisionProvider:
    """OpenAI-compatible vision provider (Responses API or Chat Completions)."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        name: str | None = None,
        wire_api: str | None = None,
        reasoning_effort: str | None = None,
        store: bool | None = None,
        system_prompt: str | None = None,
        timeout: float = 180.0,
    ) -> None:
        self.model = model or os.getenv("OPENAI_VISION_MODEL", "gpt-5.6")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.reasoning_effort = reasoning_effort or os.getenv("OPENAI_REASONING_EFFORT", "low")
        if store is None:
            self.store = os.getenv("OPENAI_STORE", "false").lower() in {"1", "true", "yes"}
        else:
            self.store = store
        self.name = name or os.getenv("OPENAI_PROVIDER_NAME", "openai-compatible")
        self.wire_api = (wire_api or os.getenv("OPENAI_WIRE_API") or "responses").strip()
        self.system_prompt = system_prompt
        self.timeout = float(timeout)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise AutoLabelError("OPENAI_API_KEY is not set")

    def _prompt_text(self, classes: list[dict[str, Any]]) -> str:
        class_lines = "\n".join(
            f"{entry['id']}: {entry['name']} (visual prompts: {', '.join(entry.get('prompts', []))})"
            for entry in classes
        )
        if self.system_prompt and self.system_prompt.strip():
            template = self.system_prompt
            if "{class_lines}" in template:
                return template.format(class_lines=class_lines)
            return f"{template.rstrip()}\n\nAllowed classes:\n{class_lines}"
        return PROMPT.format(class_lines=class_lines)

    def _payload_responses(self, image: Path, classes: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "model": self.model,
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": self._prompt_text(classes)},
                    {"type": "input_image", "image_url": prepare_image_data_url(image), "detail": "high"},
                ],
            }],
            "text": {"format": {
                "type": "json_schema", "name": "d1_detection_result",
                "strict": True, "schema": detection_schema_for(classes),
            }},
            "reasoning": {"effort": self.reasoning_effort},
            "store": self.store,
        }

    def _payload_chat(self, image: Path, classes: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            self._prompt_text(classes)
                            + "\nReturn JSON object with key detections; "
                            "each item has class_id,x_min,y_min,x_max,y_max,confidence,evidence "
                            "using 0..1000 coordinates."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": prepare_image_data_url(image)}},
                ],
            }],
            "response_format": {"type": "json_object"},
        }

    def _request(self, payload: dict[str, Any], endpoint: str) -> dict[str, Any]:
        try:
            response = httpx.post(
                f"{self.base_url}/{endpoint.lstrip('/')}",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=httpx.Timeout(self.timeout, connect=15.0),
            )
        except httpx.RequestError as exc:
            raise AutoLabelError(f"无法连接中转服务：{type(exc).__name__}") from exc
        if response.is_error:
            message = ""
            try:
                body = response.json()
                error = body.get("error", body) if isinstance(body, dict) else {}
                if isinstance(error, dict):
                    message = str(error.get("message", ""))
            except (ValueError, TypeError):
                pass
            message = re.sub(r"sk-[A-Za-z0-9_-]+", "<redacted>", message)[:300]
            suffix = f"：{message}" if message else ""
            raise AutoLabelError(f"中转返回 HTTP {response.status_code}{suffix}")
        try:
            data = response.json()
        except ValueError as exc:
            raise AutoLabelError("中转没有返回 JSON 响应") from exc
        if not isinstance(data, dict):
            raise AutoLabelError("中转响应外层不是 JSON 对象")
        return data

    def predict(self, image: Path, classes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.wire_api in {"chat", "chat_completions", "chat.completions"}:
            data = self._request(self._payload_chat(image, classes), "chat/completions")
            source_tag = f"{self.name}_chat_api"
        else:
            data = self._request(self._payload_responses(image, classes), "responses")
            source_tag = f"{self.name}_responses_api"
        try:
            parsed = DetectionResult.model_validate(
                _normalize_detections(_json_object(_response_text(data)), classes)
            )
        except ValidationError as exc:
            raise AutoLabelError(f"模型检测结果字段不符合要求：{_validation_message(exc)}") from exc
        boxes: list[dict[str, Any]] = []
        for detection in parsed.detections:
            if detection.x_max <= detection.x_min or detection.y_max <= detection.y_min:
                continue
            boxes.append({
                "class_id": detection.class_id,
                "x": ((detection.x_min + detection.x_max) / 2) / 1000,
                "y": ((detection.y_min + detection.y_max) / 2) / 1000,
                "width": (detection.x_max - detection.x_min) / 1000,
                "height": (detection.y_max - detection.y_min) / 1000,
                "confidence": detection.confidence,
                "evidence": detection.evidence,
                "source": source_tag,
            })
        return boxes


class SAM3ServiceProvider:
    """Client for the isolated local SAM 3 HTTP inference service."""

    name = "sam3-local"

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        *,
        confidence_threshold: float | None = None,
        nms_iou_threshold: float | None = None,
        timeout: float | None = None,
        name: str | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("SAM3_SERVICE_URL") or "").rstrip("/")
        self.model = model or os.getenv("SAM3_MODEL", "sam3")
        self.confidence_threshold = float(
            confidence_threshold
            if confidence_threshold is not None
            else os.getenv("SAM3_CONFIDENCE_THRESHOLD", "0.5")
        )
        self.nms_iou_threshold = float(
            nms_iou_threshold
            if nms_iou_threshold is not None
            else os.getenv("SAM3_NMS_IOU_THRESHOLD", "0.7")
        )
        self.timeout = float(
            timeout if timeout is not None else os.getenv("SAM3_REQUEST_TIMEOUT", "600")
        )
        if name:
            self.name = name
        if not self.base_url:
            raise AutoLabelError("SAM3_SERVICE_URL is not set")

    def _request(self, image: Path, classes: list[dict[str, Any]]) -> dict[str, Any]:
        content_type = mimetypes.guess_type(image.name)[0] or "application/octet-stream"
        try:
            with image.open("rb") as stream:
                response = httpx.post(
                    f"{self.base_url}/v1/predict",
                    files={"image": (image.name, stream, content_type)},
                    data={
                        "classes_json": json.dumps(classes, ensure_ascii=False),
                        "confidence_threshold": str(self.confidence_threshold),
                        "nms_iou_threshold": str(self.nms_iou_threshold),
                    },
                    timeout=httpx.Timeout(self.timeout, connect=10.0),
                )
        except httpx.RequestError as exc:
            raise AutoLabelError(f"无法连接 SAM 3 本地服务：{type(exc).__name__}") from exc
        if response.is_error:
            message = ""
            try:
                body = response.json()
                if isinstance(body, dict):
                    message = str(body.get("detail", body.get("error", "")))
            except ValueError:
                pass
            suffix = f"：{message[:300]}" if message else ""
            raise AutoLabelError(f"SAM 3 服务返回 HTTP {response.status_code}{suffix}")
        try:
            data = response.json()
        except ValueError as exc:
            raise AutoLabelError("SAM 3 服务没有返回 JSON") from exc
        if not isinstance(data, dict):
            raise AutoLabelError("SAM 3 服务响应外层不是 JSON 对象")
        return data

    def predict(self, image: Path, classes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        data = self._request(image, classes)
        width = _number(data.get("width"))
        height = _number(data.get("height"))
        detections = data.get("detections")
        if not width or not height or width <= 0 or height <= 0:
            raise AutoLabelError("SAM 3 服务返回了无效图片尺寸")
        if not isinstance(detections, list):
            raise AutoLabelError("SAM 3 服务响应缺少 detections 数组")
        boxes: list[dict[str, Any]] = []
        for index, detection in enumerate(detections):
            if not isinstance(detection, dict):
                raise AutoLabelError(f"SAM 3 第 {index + 1} 个检测结果不是对象")
            class_id = _class_id(detection.get("class_id"), classes)
            coords = _coordinates(detection)
            if class_id is None or coords is None:
                raise AutoLabelError(f"SAM 3 第 {index + 1} 个检测结果缺少类别或坐标")
            x_min, y_min, x_max, y_max = coords
            x_min, x_max = max(0.0, x_min), min(width, x_max)
            y_min, y_max = max(0.0, y_min), min(height, y_max)
            if x_max <= x_min or y_max <= y_min:
                continue
            confidence = _number(detection.get("confidence", detection.get("score")))
            boxes.append({
                "class_id": class_id,
                "x": ((x_min + x_max) / 2) / width,
                "y": ((y_min + y_max) / 2) / height,
                "width": (x_max - x_min) / width,
                "height": (y_max - y_min) / height,
                "confidence": confidence if confidence is not None else 0.5,
                "evidence": str(detection.get("prompt", detection.get("evidence", "sam3"))),
                "source": "sam3-local",
            })
        if isinstance(data.get("model"), str) and data["model"]:
            self.model = data["model"]
        return boxes


def provider_status() -> dict[str, Any]:
    return {
        "provider": os.getenv("OPENAI_PROVIDER_NAME", "openai-compatible"),
        "model": os.getenv("OPENAI_VISION_MODEL", "gpt-5.6"),
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "wire_api": "responses",
        "reasoning_effort": os.getenv("OPENAI_REASONING_EFFORT", "low"),
        "store": os.getenv("OPENAI_STORE", "false").lower() in {"1", "true", "yes"},
        "configured": bool(os.getenv("OPENAI_API_KEY")),
    }


def sam3_provider_status() -> dict[str, Any]:
    return {
        "id": "sam3",
        "provider": "sam3-local",
        "name": "SAM 3 本地模型",
        "model": os.getenv("SAM3_MODEL", "sam3"),
        "base_url": os.getenv("SAM3_SERVICE_URL", ""),
        "configured": bool(os.getenv("SAM3_SERVICE_URL")),
        "confidence_threshold": float(os.getenv("SAM3_CONFIDENCE_THRESHOLD", "0.5")),
        "nms_iou_threshold": float(os.getenv("SAM3_NMS_IOU_THRESHOLD", "0.7")),
    }


def provider_catalog(
    injected: AutoLabelProvider | None = None,
    settings_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    if injected is not None:
        return [{
            "id": "openai",
            "provider": injected.name,
            "name": injected.name,
            "model": injected.model,
            "configured": True,
        }]
    catalog: list[dict[str, Any]] = []
    # Legacy env-based entries remain available.
    openai = {"id": "openai", "name": "视觉大模型(环境变量)", **provider_status()}
    catalog.append(openai)
    catalog.append(sam3_provider_status())
    # Web-configured profiles (preferred for new tasks).
    try:
        from .settings_store import SettingsStore, default_settings_dir

        store = SettingsStore(Path(settings_dir) if settings_dir else default_settings_dir())
        for profile in store.list_profiles():
            if not profile.get("enabled", True):
                continue
            pid = str(profile["id"])
            if profile.get("type") == "sam3":
                catalog.append({
                    "id": pid,
                    "provider": "sam3-local",
                    "name": profile.get("name") or pid,
                    "model": profile.get("model", "sam3"),
                    "base_url": profile.get("service_url", ""),
                    "configured": bool(profile.get("service_url")),
                    "source": "settings",
                })
            else:
                key_ok = bool(store.get_key_secret(str(profile.get("key_id") or ""))) or bool(
                    os.getenv("OPENAI_API_KEY")
                )
                catalog.append({
                    "id": pid,
                    "provider": profile.get("name") or pid,
                    "name": profile.get("name") or pid,
                    "model": profile.get("model", ""),
                    "base_url": profile.get("base_url", ""),
                    "wire_api": profile.get("wire_api", "responses"),
                    "configured": bool(profile.get("base_url") and profile.get("model") and key_ok),
                    "source": "settings",
                })
    except Exception:
        # Settings are optional; env providers still work.
        pass
    # De-duplicate by id keeping first occurrence (env aliases first).
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in catalog:
        if item["id"] in seen:
            continue
        seen.add(item["id"])
        unique.append(item)
    return unique


def create_provider_from_profile(
    profile: dict[str, Any],
    *,
    api_key: str | None = None,
    system_prompt: str | None = None,
    confidence_threshold: float | None = None,
) -> AutoLabelProvider:
    ptype = str(profile.get("type") or "openai_compatible")
    if ptype == "sam3":
        return SAM3ServiceProvider(
            base_url=str(profile.get("service_url") or ""),
            model=str(profile.get("model") or "sam3"),
            confidence_threshold=(
                confidence_threshold
                if confidence_threshold is not None
                else float(profile.get("confidence_threshold", 0.5))
            ),
            nms_iou_threshold=float(profile.get("nms_iou_threshold", 0.7)),
            timeout=float(profile.get("request_timeout_sec", 600)),
            name=str(profile.get("name") or profile.get("id") or "sam3-local"),
        )
    return OpenAIVisionProvider(
        model=str(profile.get("model") or ""),
        api_key=api_key,
        base_url=str(profile.get("base_url") or ""),
        name=str(profile.get("name") or profile.get("id") or "openai-compatible"),
        wire_api=str(profile.get("wire_api") or "responses"),
        reasoning_effort=str(profile.get("reasoning_effort") or "medium"),
        store=bool(profile.get("store", False)),
        system_prompt=system_prompt,
        timeout=float(profile.get("request_timeout_sec", 180)),
    )


def create_provider(
    provider_id: str,
    *,
    settings_dir: Path | str | None = None,
    system_prompt: str | None = None,
    key_id: str | None = None,
    confidence_threshold: float | None = None,
) -> AutoLabelProvider:
    # Backward-compatible env providers.
    if provider_id == "openai":
        return OpenAIVisionProvider(system_prompt=system_prompt)
    if provider_id == "sam3":
        return SAM3ServiceProvider(
            confidence_threshold=confidence_threshold,
        )

    from .settings_store import SettingsStore, default_settings_dir

    store = SettingsStore(Path(settings_dir) if settings_dir else default_settings_dir())
    profile = store.get_profile(provider_id)
    if profile is None:
        raise AutoLabelError(f"未知自动标注模型：{provider_id}")
    secret = None
    if profile.get("type") != "sam3":
        use_key = key_id or str(profile.get("key_id") or "")
        secret = store.get_key_secret(use_key) if use_key else None
        if not secret:
            secret = os.getenv("OPENAI_API_KEY")
        if not secret:
            raise AutoLabelError(f"模型档案 {provider_id} 未绑定可用 API Key")
    return create_provider_from_profile(
        profile,
        api_key=secret,
        system_prompt=system_prompt,
        confidence_threshold=confidence_threshold,
    )
