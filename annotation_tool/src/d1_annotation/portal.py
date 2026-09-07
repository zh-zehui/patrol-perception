from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .datasets import DatasetSource, load_registry, save_registry
from .settings_store import SettingsStore, default_settings_dir
from .autolabel import PROMPT as DEFAULT_AUTOLABEL_PROMPT

STATIC_DIR = Path(__file__).with_name("static")
DEFAULT_CLASSES = Path(__file__).resolve().parents[2] / "config" / "classes.yaml"


@dataclass
class RunningDataset:
    port: int
    process: subprocess.Popen


class RegisterRequest(BaseModel):
    id: str
    name: str = ""
    path: str
    notes: str = ""
    model_profile_id: str = ""
    key_id: str = ""
    system_prompt: str = ""
    confidence_threshold: float = 0.5
    classes: list[dict[str, Any]] = Field(default_factory=list)
    source_type: str = "images_dataset"
    stream: dict[str, Any] = Field(default_factory=dict)


class StartRequest(BaseModel):
    rescan: bool = False


class KeyUpsertRequest(BaseModel):
    id: str
    name: str = ""
    api_key: str = ""
    enabled: bool = True
    note: str = ""


class ProfileUpsertRequest(BaseModel):
    id: str
    name: str = ""
    type: str = "openai_compatible"
    enabled: bool = True
    note: str = ""
    base_url: str = ""
    wire_api: str = "responses"
    model: str = ""
    key_id: str = ""
    reasoning_effort: str = "medium"
    store: bool = False
    service_url: str = "http://127.0.0.1:8783"
    confidence_threshold: float = 0.5
    nms_iou_threshold: float = 0.7
    request_timeout_sec: float = 180
    make_default: bool = False


class DefaultProfileRequest(BaseModel):
    default_profile_id: str = ""


class DatasetPortal:
    def __init__(
        self,
        registry: Path,
        classes: Path,
        bind_host: str = "0.0.0.0",
        public_host: str = "",
        first_port: int = 8800,
        settings_dir: Path | None = None,
    ) -> None:
        self.registry = registry.expanduser().resolve()
        self.classes = classes.expanduser().resolve()
        self.bind_host = bind_host
        self.public_host = public_host
        self.first_port = first_port
        self.settings = SettingsStore(settings_dir or default_settings_dir(self.registry))
        self._running: dict[str, RunningDataset] = {}
        self._lock = threading.Lock()

    def sources(self) -> list[DatasetSource]:
        return load_registry(self.registry)

    def source(self, dataset_id: str) -> DatasetSource:
        source = next((item for item in self.sources() if item.id == dataset_id), None)
        if source is None:
            raise HTTPException(status_code=404, detail="数据集未注册")
        return source

    def resolve_classes_path(self, dataset_id: str) -> Path:
        task = self.settings.get_task(dataset_id)
        if task and task.get("classes_file"):
            path = Path(str(task["classes_file"]))
            if path.is_file():
                return path
        local = self.settings.classes_path(dataset_id)
        if local.is_file():
            return local
        return self.classes

    @staticmethod
    def _port_available(port: int) -> bool:
        with socket.socket() as sock:
            return sock.connect_ex(("127.0.0.1", port)) != 0

    def _allocate_port(self) -> int:
        used = {item.port for item in self._running.values()}
        for port in range(self.first_port, self.first_port + 1000):
            if port not in used and self._port_available(port):
                return port
        raise HTTPException(status_code=503, detail="没有可用的标注服务端口")

    def start(self, dataset_id: str, rescan: bool = False) -> int:
        source = self.source(dataset_id)
        with self._lock:
            active = self._running.get(dataset_id)
            if active and active.process.poll() is None:
                return active.port
            port = self._allocate_port()
            classes_path = self.resolve_classes_path(dataset_id)
            task = self.settings.get_task(dataset_id) or {}
            command = [
                sys.executable,
                "-m",
                "d1_annotation.cli",
                "--dataset",
                str(source.root),
                "--classes",
                str(classes_path),
                "--registry",
                str(self.registry),
                "--settings-dir",
                str(self.settings.root),
                "--task-id",
                dataset_id,
                "--host",
                self.bind_host,
                "--port",
                str(port),
            ]
            if rescan:
                command.append("--rescan")
            env = os.environ.copy()
            if task.get("model_profile_id"):
                env["D1_DEFAULT_PROVIDER_ID"] = str(task["model_profile_id"])
            if task.get("system_prompt"):
                env["D1_TASK_SYSTEM_PROMPT"] = str(task["system_prompt"])
            if task.get("key_id"):
                env["D1_TASK_KEY_ID"] = str(task["key_id"])
            if task.get("confidence_threshold") is not None:
                env["D1_TASK_CONFIDENCE"] = str(task["confidence_threshold"])
            process = subprocess.Popen(command, env=env)
            self._running[dataset_id] = RunningDataset(port, process)
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    self._running.pop(dataset_id, None)
                    raise HTTPException(status_code=502, detail="标注服务启动失败，请查看服务器终端日志")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                        return port
                except OSError:
                    time.sleep(0.1)
            raise HTTPException(status_code=504, detail="标注服务启动超时，请稍后重试")

    def url(self, port: int, request_host: str) -> str:
        host = self.public_host or request_host.split(":", 1)[0]
        return f"http://{host}:{port}/"

    def stop(self, dataset_id: str) -> bool:
        with self._lock:
            active = self._running.pop(dataset_id, None)
        if active is None:
            return False
        if active.process.poll() is None:
            active.process.terminate()
            try:
                active.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                active.process.kill()
                active.process.wait(timeout=5)
        return True

    def reset_annotations(self, dataset_id: str) -> dict[str, Any]:
        source = self.source(dataset_id)
        root = Path(source.root).expanduser().resolve()
        if not root.is_dir():
            raise HTTPException(status_code=422, detail="数据集路径无效或不存在")
        stopped = self.stop(dataset_id)
        deleted: list[str] = []
        for name in (".d1_annotations", "labels"):
            target = (root / name).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="路径校验失败") from exc
            if not target.exists():
                continue
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            deleted.append(name)
        return {"ok": True, "deleted": deleted, "stopped": stopped, "root": str(root)}

    def default_classes_template(self) -> list[dict[str, Any]]:
        if self.classes.is_file():
            raw = yaml.safe_load(self.classes.read_text(encoding="utf-8")) or {}
            classes = raw.get("classes")
            if isinstance(classes, list) and classes:
                return classes
        return [
            {"id": 0, "name": "object", "color": "#22C55E", "prompts": ["object"]},
        ]


def create_portal_app(portal: DatasetPortal) -> FastAPI:
    app = FastAPI(title="D1 Annotation Portal", version="0.3.0")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        page = STATIC_DIR / "portal.html"
        if page.is_file():
            return FileResponse(page)
        return HTMLResponse("<h1>缺少 static/portal.html</h1>", status_code=500)

    @app.get("/api/portal/settings")
    def get_settings():
        return portal.settings.public_snapshot()

    @app.get("/api/portal/class-template")
    def class_template():
        return {"classes": portal.default_classes_template()}

    @app.get("/api/portal/default-prompt")
    def default_prompt():
        """Expose autolabel.PROMPT so the Web form can show/edit the built-in template."""
        return {
            "prompt": DEFAULT_AUTOLABEL_PROMPT,
            "note": "留空保存时仍会使用该内置模板；页面展示便于查看和修改。",
        }

    @app.post("/api/portal/keys")
    def upsert_key(request: KeyUpsertRequest):
        try:
            return {"ok": True, "key": portal.settings.upsert_key(request.model_dump())}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.delete("/api/portal/keys/{key_id}")
    def delete_key(key_id: str):
        portal.settings.delete_key(key_id)
        return {"ok": True}

    @app.post("/api/portal/profiles")
    def upsert_profile(request: ProfileUpsertRequest):
        try:
            profile = portal.settings.upsert_profile(
                request.model_dump(exclude={"make_default"}),
                make_default=request.make_default,
            )
            return {"ok": True, "profile": profile}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.delete("/api/portal/profiles/{profile_id}")
    def delete_profile(profile_id: str):
        portal.settings.delete_profile(profile_id)
        return {"ok": True}

    @app.put("/api/portal/profiles/default")
    def set_default_profile(request: DefaultProfileRequest):
        profiles = portal.settings.list_profiles()
        try:
            portal.settings.save_profiles(profiles, request.default_profile_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"ok": True, "default_profile_id": portal.settings.default_profile_id()}

    @app.get("/api/portal/datasets")
    def datasets():
        tasks = {t["id"]: t for t in portal.settings.list_tasks()}
        rows = []
        for source in portal.sources():
            task = tasks.get(source.id, {})
            rows.append({
                "id": source.id,
                "name": source.name,
                "path": str(source.root),
                "notes": task.get("notes", ""),
                "model_profile_id": task.get("model_profile_id", ""),
                "has_task_config": bool(task),
                "class_count": (
                    len(yaml.safe_load(Path(task["classes_file"]).read_text(encoding="utf-8")).get("classes", []))
                    if task.get("classes_file") and Path(str(task["classes_file"])).is_file()
                    else None
                ),
            })
        return {"datasets": rows, "settings_dir": str(portal.settings.root)}

    @app.get("/api/portal/datasets/{dataset_id}")
    def get_dataset(dataset_id: str):
        source = portal.source(dataset_id)
        task = portal.settings.get_task(dataset_id) or {
            "id": source.id,
            "name": source.name,
            "path": str(source.root),
            "notes": "",
            "model_profile_id": portal.settings.default_profile_id(),
            "key_id": "",
            "system_prompt": "",
            "confidence_threshold": 0.5,
            "source_type": "images_dataset",
            "stream": {},
        }
        classes_path = portal.resolve_classes_path(dataset_id)
        classes = []
        if classes_path.is_file():
            classes = (yaml.safe_load(classes_path.read_text(encoding="utf-8")) or {}).get("classes") or []
        return {"dataset": {"id": source.id, "name": source.name, "path": str(source.root)}, "task": task, "classes": classes}

    @app.post("/api/portal/datasets")
    def register(request: RegisterRequest):
        if not request.id.strip() or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for c in request.id
        ):
            raise HTTPException(status_code=422, detail="数据集 ID 只能包含字母、数字、下划线和连字符")
        sources = portal.sources()
        if any(s.id == request.id for s in sources):
            raise HTTPException(status_code=409, detail="数据集 ID 已存在")
        root = Path(request.path).expanduser().resolve()
        if request.source_type == "images_dataset" and not (root / "images").is_dir():
            raise HTTPException(status_code=422, detail="数据集路径不存在或缺少 images/ 目录")
        if not root.exists():
            raise HTTPException(status_code=422, detail="路径不存在")
        classes = request.classes or portal.default_classes_template()
        try:
            portal.settings.save_task(request.model_dump(), classes)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        source = DatasetSource(request.id, request.name or request.id, root, read_only=False)
        sources.append(source)
        save_registry(portal.registry, sources)
        return {"ok": True, "id": source.id}

    @app.put("/api/portal/datasets/{dataset_id}")
    def update_dataset(dataset_id: str, request: RegisterRequest):
        source = portal.source(dataset_id)
        if request.id != dataset_id:
            raise HTTPException(status_code=422, detail="路径参数与任务 ID 不一致")
        root = Path(request.path or source.root).expanduser().resolve()
        if request.source_type == "images_dataset" and not (root / "images").is_dir():
            raise HTTPException(status_code=422, detail="数据集路径不存在或缺少 images/ 目录")
        classes = request.classes or portal.default_classes_template()
        payload = request.model_dump()
        payload["path"] = str(root)
        try:
            portal.settings.save_task(payload, classes)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        sources = portal.sources()
        updated = []
        for item in sources:
            if item.id == dataset_id:
                updated.append(
                    DatasetSource(item.id, request.name or item.name, root, item.source_type, item.read_only, item.exhaustive_classes)
                )
            else:
                updated.append(item)
        save_registry(portal.registry, updated)
        return {"ok": True, "id": dataset_id}

    @app.delete("/api/portal/datasets/{dataset_id}")
    def delete_dataset(dataset_id: str):
        portal.source(dataset_id)
        sources = [s for s in portal.sources() if s.id != dataset_id]
        save_registry(portal.registry, sources)
        portal.settings.delete_task(dataset_id)
        return {"ok": True}

    @app.post("/api/portal/datasets/{dataset_id}/start")
    def start(dataset_id: str, start_request: StartRequest, http_request: Request):
        port = portal.start(dataset_id, start_request.rescan)
        return {
            "ok": True,
            "port": port,
            "url": portal.url(port, http_request.headers.get("host", "127.0.0.1")),
        }

    @app.post("/api/portal/datasets/{dataset_id}/reset-annotations")
    def reset_annotations(dataset_id: str):
        return portal.reset_annotations(dataset_id)

    return app
