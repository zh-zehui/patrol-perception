#!/usr/bin/env python3
"""Capture the real local UI with temporary synthetic data and no API credentials."""
from __future__ import annotations

import os
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen

from create_demo_dataset import create_demo_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "annotation_tool/src"))


def main():
    from d1_annotation.core import YoloDataset, load_classes
    from playwright.sync_api import sync_playwright
    import yaml

    out = ROOT / "docs/assets"
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="patrol-showcase-") as tmp:
        work = Path(tmp)
        dataset = create_demo_dataset(work / "synthetic_demo")
        classes = ROOT / "annotation_tool/config/classes.yaml"
        ds = YoloDataset(dataset, load_classes(classes))
        ds.save_draft(0, ds.read_boxes(0))
        ds.set_review(0, "draft_ready")
        ds.set_review(1, "reviewed")
        ds.set_review(2, "needs_review")
        registry = work / "datasets_registry.yaml"
        registry.write_text(yaml.safe_dump({"version": 1, "datasets": [{"id": "synthetic_demo", "name": "Synthetic UI demonstration", "path": str(dataset), "source_type": "self_collected", "read_only": True}]}))
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env = {k: v for k, v in os.environ.items() if not k.startswith(("OPENAI_", "SAM3_", "D1_", "PATROL_", "HF_", "GH_", "GITHUB_"))}
        env["D1_AUTOLABEL_SETTINGS_DIR"] = str(work / "settings")
        env["PYTHONPATH"] = str(ROOT / "annotation_tool/src")
        cmd = [sys.executable, "-m", "d1_annotation.cli", "--dataset", str(dataset), "--classes", str(classes), "--registry", str(registry), "--settings-dir", str(work / "settings"), "--host", "127.0.0.1", "--port", str(port)]
        with (work / "server.log").open("w+") as log:
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=log)
            try:
                address = f"http://127.0.0.1:{port}"
                deadline = time.monotonic() + 20
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("Temporary annotation service exited before startup")
                    try:
                        with urlopen(address + "/api/config", timeout=1) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise RuntimeError("Temporary annotation service startup timed out")
                        time.sleep(.1)
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True, executable_path=shutil.which("chromium"))
                    try:
                        page = browser.new_page(viewport={"width": 1920, "height": 1120}, device_scale_factor=1)
                        errors = []
                        page.on("pageerror", lambda exc: errors.append(str(exc)))
                        page.goto(address, wait_until="networkidle")
                        page.wait_for_function("document.querySelector('#imageList').children.length >= 3")
                        page.wait_for_function("document.querySelector('#baselineList').textContent.includes('person')")
                        page.evaluate("document.fonts.ready")
                        page.screenshot(path=str(out / "annotation-demo.png"))
                        page.goto(address + "/builder", wait_until="networkidle")
                        page.locator("#refreshDatasets").click()
                        page.wait_for_function("document.querySelector('#message').textContent === '扫描完成'")
                        # Select the three fixture classes through the actual controls.
                        for checkbox in page.locator("#classes input").all():
                            checkbox.set_checked(checkbox.get_attribute("value") in {"0", "2", "7"})
                        page.evaluate("document.fonts.ready")
                        page.screenshot(path=str(out / "dataset-builder.png"))
                        if errors:
                            raise RuntimeError(f"UI JavaScript errors: {errors}")
                    finally:
                        browser.close()
            finally:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
    print("Captured annotation editor and dataset builder using isolated synthetic fixtures.")


if __name__ == "__main__":
    main()
