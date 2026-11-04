#!/usr/bin/env python3
"""Check tracked files for common secrets, local data and oversized artifacts.

Only file names are printed on failure; matched credential values are never shown.
Use --history before the first publication to scan every reachable Git blob.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = [
    re.compile(rb"(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,})"),
    re.compile(rb"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----"),
    re.compile(rb"https?://[^\s/:]+:[^\s/@]+@github\.com"),
]
MAX_BYTES = 5 * 1024 * 1024


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def forbidden(path: str) -> bool:
    parts = Path(path).parts
    name = Path(path).name
    return (
        name == ".hf_token"
        or (name.startswith(".env") and name != ".env.example")
        or path == "annotation_tool/config/datasets_registry.yaml"
        or path.startswith(("annotation_tool/config/local/", "datasets/logs/", "data/demo_dataset/", "data/weights/", "data/eval_runs/", "data/runs/", "datasets/downloads/"))
        or any(p in {".venv", "__pycache__", ".d1_annotations"} for p in parts)
        or Path(path).suffix.lower() in {".pt", ".pth", ".onnx", ".engine", ".trt", ".safetensors", ".ckpt", ".bag", ".db3", ".mcap", ".pem", ".key", ".log"}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args()
    errors: list[str] = []
    paths = [p.decode() for p in git("ls-files", "-z").split(b"\0") if p]
    for path in paths:
        if forbidden(path):
            errors.append(f"Excluded content is tracked: {path}")
        file = ROOT / path
        if not file.is_file():
            continue
        data = file.read_bytes()
        if len(data) > MAX_BYTES:
            errors.append(f"File exceeds 5 MiB: {path}")
        if any(p.search(data) for p in PATTERNS):
            errors.append(f"Possible credential: {path}")
    if args.history:
        # Inspect all blobs, including deleted/renamed files. Do not print values.
        objects = git("rev-list", "--objects", "--all").splitlines()
        proc = subprocess.run(["git", "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"], cwd=ROOT,
                              input=b"\n".join(row.split(b" ", 1)[0] for row in objects), stdout=subprocess.PIPE, check=True)
        names = {row.split(b" ", 1)[0]: row.split(b" ", 1)[1].decode(errors="replace") if b" " in row else "" for row in objects}
        for row in proc.stdout.splitlines():
            oid, kind, size = row.split()
            if kind != b"blob":
                continue
            if int(size) > MAX_BYTES:
                errors.append(f"Oversized historical blob: {names.get(oid, oid.decode())}")
            blob = git("cat-file", "blob", oid.decode())
            if any(p.search(blob) for p in PATTERNS):
                errors.append(f"Possible historical credential: {names.get(oid, oid.decode())}")
    if errors:
        for error in sorted(set(errors)):
            print(error)
        raise SystemExit(1)
    print(f"Publication check passed: {len(paths)} tracked files" + ("; reachable history scanned" if args.history else ""))


if __name__ == "__main__":
    main()
