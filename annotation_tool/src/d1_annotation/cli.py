from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from .app import create_app

DEFAULT_CLASSES = Path(__file__).resolve().parents[2] / "config" / "classes.yaml"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Review and edit a YOLO detection dataset in a browser")
    value.add_argument("--dataset", required=True, type=Path, help="YOLO dataset root containing images/ and labels/")
    value.add_argument("--classes", type=Path, default=DEFAULT_CLASSES, help="D1 class configuration YAML")
    value.add_argument("--host", default="127.0.0.1", help="Bind address; use 0.0.0.0 only on a trusted LAN")
    value.add_argument("--port", default=8765, type=int)
    value.add_argument("--reload", action="store_true", help="Developer mode")
    value.add_argument("--registry", type=Path, default=None, help="YAML registry of additional datasets")
    value.add_argument("--rescan", action="store_true", help="重新扫描主数据集图片并更新 image_index.json")
    value.add_argument("--settings-dir", type=Path, default=None, help="Web settings dir (keys/profiles/tasks)")
    value.add_argument("--task-id", default="", help="Portal task id for provider/prompt defaults")
    return value


def main() -> None:
    args = parser().parse_args()
    settings_dir = args.settings_dir
    if settings_dir is None and os.getenv("D1_AUTOLABEL_SETTINGS_DIR"):
        settings_dir = Path(os.environ["D1_AUTOLABEL_SETTINGS_DIR"])
    app = create_app(
        args.dataset,
        args.classes,
        registry_path=args.registry,
        rescan=args.rescan,
        settings_dir=settings_dir,
        task_id=args.task_id or os.getenv("D1_TASK_ID", ""),
    )
    print(f"Dataset: {args.dataset.expanduser().resolve()}")
    print(f"Open: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
