from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def default_settings_dir(registry_path: Path | None = None) -> Path:
    env = os.getenv("D1_AUTOLABEL_SETTINGS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    if registry_path is not None:
        return registry_path.expanduser().resolve().parent / "local"
    return Path(__file__).resolve().parents[2] / "config" / "local"


class SettingsStore:
    """Web-editable keys, model profiles, and per-task annotation settings.

    Secrets live under config/local/ (gitignored). API keys are never returned
    in full by public helpers — only last-4 mask.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.keys_path = self.root / "keys.yaml"
        self.profiles_path = self.root / "model_profiles.yaml"
        self.tasks_dir = self.root / "tasks"
        self.classes_dir = self.root / "classes"
        self.root.mkdir(parents=True, exist_ok=True)
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.classes_dir.mkdir(parents=True, exist_ok=True)

    def _read(self, path: Path, default: dict[str, Any]) -> dict[str, Any]:
        if not path.exists():
            return dict(default)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"invalid yaml mapping: {path}")
        return raw

    def _write(self, path: Path, payload: dict[str, Any], private: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        if private:
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass

    @staticmethod
    def _mask_key(value: str) -> str:
        value = value or ""
        if len(value) <= 4:
            return "****" if value else ""
        return f"...{value[-4:]}"

    def list_keys(self, *, reveal: bool = False) -> list[dict[str, Any]]:
        raw = self._read(self.keys_path, {"keys": []})
        items = raw.get("keys") or []
        result = []
        for item in items:
            if not isinstance(item, dict):
                continue
            api_key = str(item.get("api_key", ""))
            entry = {
                "id": str(item.get("id", "")),
                "name": str(item.get("name", item.get("id", ""))),
                "enabled": bool(item.get("enabled", True)),
                "note": str(item.get("note", "")),
                "configured": bool(api_key),
                "api_key_masked": self._mask_key(api_key),
            }
            if reveal:
                entry["api_key"] = api_key
            result.append(entry)
        return result

    def upsert_key(self, payload: dict[str, Any]) -> dict[str, Any]:
        key_id = str(payload.get("id", "")).strip()
        if not key_id or not ID_RE.match(key_id):
            raise ValueError("Key ID 只能包含字母、数字、下划线和连字符")
        raw = self._read(self.keys_path, {"keys": []})
        keys = [x for x in (raw.get("keys") or []) if isinstance(x, dict)]
        existing = next((x for x in keys if str(x.get("id")) == key_id), None)
        api_key = str(payload.get("api_key", "")).strip()
        if existing is None and not api_key:
            raise ValueError("新建 Key 必须填写 api_key")
        if existing is not None and not api_key:
            api_key = str(existing.get("api_key", ""))
        entry = {
            "id": key_id,
            "name": str(payload.get("name") or key_id),
            "api_key": api_key,
            "enabled": bool(payload.get("enabled", True)),
            "note": str(payload.get("note", "")),
        }
        keys = [x for x in keys if str(x.get("id")) != key_id] + [entry]
        self._write(self.keys_path, {"version": 1, "keys": keys}, private=True)
        return self.list_keys()[-1] if False else next(
            x for x in self.list_keys() if x["id"] == key_id
        )

    def delete_key(self, key_id: str) -> None:
        raw = self._read(self.keys_path, {"keys": []})
        keys = [x for x in (raw.get("keys") or []) if isinstance(x, dict) and str(x.get("id")) != key_id]
        self._write(self.keys_path, {"version": 1, "keys": keys}, private=True)

    def get_key_secret(self, key_id: str) -> str | None:
        for item in self._read(self.keys_path, {"keys": []}).get("keys") or []:
            if isinstance(item, dict) and str(item.get("id")) == key_id and item.get("enabled", True):
                secret = str(item.get("api_key", "")).strip()
                return secret or None
        return None

    def list_profiles(self) -> list[dict[str, Any]]:
        raw = self._read(self.profiles_path, {"profiles": [], "default_profile_id": ""})
        profiles = []
        for item in raw.get("profiles") or []:
            if isinstance(item, dict) and item.get("id"):
                profiles.append(dict(item))
        return profiles

    def default_profile_id(self) -> str:
        raw = self._read(self.profiles_path, {"profiles": [], "default_profile_id": ""})
        return str(raw.get("default_profile_id") or "")

    def save_profiles(self, profiles: list[dict[str, Any]], default_profile_id: str = "") -> None:
        cleaned = []
        seen = set()
        for item in profiles:
            if not isinstance(item, dict):
                continue
            pid = str(item.get("id", "")).strip()
            if not pid or not ID_RE.match(pid):
                raise ValueError(f"非法模型档案 ID: {pid}")
            if pid in seen:
                raise ValueError(f"重复模型档案 ID: {pid}")
            seen.add(pid)
            ptype = str(item.get("type", "openai_compatible"))
            entry = {
                "id": pid,
                "name": str(item.get("name") or pid),
                "type": ptype,
                "enabled": bool(item.get("enabled", True)),
                "note": str(item.get("note", "")),
            }
            if ptype == "sam3":
                entry.update({
                    "service_url": str(item.get("service_url") or "http://127.0.0.1:8783"),
                    "model": str(item.get("model") or "sam3"),
                    "confidence_threshold": float(item.get("confidence_threshold", 0.5)),
                    "nms_iou_threshold": float(item.get("nms_iou_threshold", 0.7)),
                    "request_timeout_sec": float(item.get("request_timeout_sec", 600)),
                })
            else:
                entry.update({
                    "base_url": str(item.get("base_url") or "").rstrip("/"),
                    "wire_api": str(item.get("wire_api") or "responses"),
                    "model": str(item.get("model") or ""),
                    "key_id": str(item.get("key_id") or ""),
                    "reasoning_effort": str(item.get("reasoning_effort") or "medium"),
                    "store": bool(item.get("store", False)),
                    "request_timeout_sec": float(item.get("request_timeout_sec", 180)),
                })
                if not entry["base_url"] or not entry["model"]:
                    raise ValueError(f"模型档案 {pid} 需要 base_url 和 model")
            cleaned.append(entry)
        if default_profile_id and default_profile_id not in seen and cleaned:
            raise ValueError("default_profile_id 不存在")
        self._write(
            self.profiles_path,
            {
                "version": 1,
                "default_profile_id": default_profile_id or (cleaned[0]["id"] if cleaned else ""),
                "profiles": cleaned,
            },
        )

    def upsert_profile(self, payload: dict[str, Any], make_default: bool = False) -> dict[str, Any]:
        profiles = self.list_profiles()
        pid = str(payload.get("id", "")).strip()
        profiles = [p for p in profiles if str(p.get("id")) != pid] + [payload]
        default_id = pid if make_default else self.default_profile_id()
        self.save_profiles(profiles, default_id)
        return next(p for p in self.list_profiles() if p["id"] == pid)

    def delete_profile(self, profile_id: str) -> None:
        profiles = [p for p in self.list_profiles() if str(p.get("id")) != profile_id]
        default_id = self.default_profile_id()
        if default_id == profile_id:
            default_id = profiles[0]["id"] if profiles else ""
        self.save_profiles(profiles, default_id)

    def get_profile(self, profile_id: str) -> dict[str, Any] | None:
        return next((p for p in self.list_profiles() if p["id"] == profile_id and p.get("enabled", True)), None)

    def task_path(self, task_id: str) -> Path:
        return self.tasks_dir / f"{task_id}.yaml"

    def classes_path(self, task_id: str) -> Path:
        return self.classes_dir / f"{task_id}.yaml"

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        path = self.task_path(task_id)
        if not path.exists():
            return None
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return raw if isinstance(raw, dict) else None

    def list_tasks(self) -> list[dict[str, Any]]:
        items = []
        for path in sorted(self.tasks_dir.glob("*.yaml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if isinstance(raw, dict) and raw.get("id"):
                items.append(raw)
        return items

    def save_task(self, payload: dict[str, Any], classes: list[dict[str, Any]]) -> dict[str, Any]:
        task_id = str(payload.get("id", "")).strip()
        if not task_id or not ID_RE.match(task_id):
            raise ValueError("任务 ID 只能包含字母、数字、下划线和连字符")
        if not classes:
            raise ValueError("至少配置一个类别")
        ids = [int(c["id"]) for c in classes]
        if ids != list(range(len(classes))):
            raise ValueError("类别 ID 必须从 0 连续递增")
        names = [str(c.get("name", "")) for c in classes]
        if len(set(names)) != len(names) or any(not n for n in names):
            raise ValueError("类别 name 必须非空且唯一")

        classes_file = self.classes_path(task_id)
        self._write(classes_file, {"classes": classes})

        task = {
            "id": task_id,
            "name": str(payload.get("name") or task_id),
            "path": str(payload.get("path") or ""),
            "notes": str(payload.get("notes") or ""),
            "model_profile_id": str(payload.get("model_profile_id") or self.default_profile_id()),
            "key_id": str(payload.get("key_id") or ""),
            "system_prompt": str(payload.get("system_prompt") or ""),
            "confidence_threshold": float(payload.get("confidence_threshold", 0.5)),
            "classes_file": str(classes_file),
            "source_type": str(payload.get("source_type") or "images_dataset"),
            # reserved for live later
            "stream": payload.get("stream") or {},
        }
        self._write(self.task_path(task_id), task)
        return task

    def delete_task(self, task_id: str) -> None:
        for path in (self.task_path(task_id), self.classes_path(task_id)):
            if path.exists():
                path.unlink()

    def public_snapshot(self) -> dict[str, Any]:
        return {
            "settings_dir": str(self.root),
            "keys": self.list_keys(reveal=False),
            "profiles": self.list_profiles(),
            "default_profile_id": self.default_profile_id(),
            "tasks": self.list_tasks(),
        }
