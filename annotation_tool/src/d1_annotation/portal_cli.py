from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .cli import DEFAULT_CLASSES
from .portal import DatasetPortal, create_portal_app
from .settings_store import default_settings_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="D1 multi-dataset annotation portal")
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--first-dataset-port", default=8800, type=int)
    parser.add_argument("--public-host", default="", help="Hostname/IP shown to browser clients")
    parser.add_argument("--settings-dir", type=Path, default=None)
    args = parser.parse_args()
    settings_dir = args.settings_dir or default_settings_dir(args.registry)
    portal = DatasetPortal(
        args.registry,
        args.classes,
        args.host,
        args.public_host,
        args.first_dataset_port,
        settings_dir=settings_dir,
    )
    print(f"Portal settings dir: {settings_dir}")
    uvicorn.run(create_portal_app(portal), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
