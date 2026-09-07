#!/usr/bin/env python3
"""Smoke-test YOLO26 detect weights (n/s/m/l/x) on GPU.

Usage (from repo, after activating pytorch_cu130):
  source /home/sribd/D1_EDU/Pytorch/switch_cuda_mode.sh 1
  cd /home/sribd/D1_EDU/patrol_perception
  python scripts/smoke_yolo26_weights.py --sizes n,s

Source image and annotated results are saved under data/weights/.
"""

from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = ROOT / "data" / "weights"
# Official sample from ultralytics/assets (stable direct link)
DEFAULT_SOURCE = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
    "gemini-image1.jpg"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sizes",
        default="n,s,m,l,x",
        help="Comma-separated YOLO26 sizes to test (default: n,s,m,l,x)",
    )
    p.add_argument("--source", default=DEFAULT_SOURCE, help="Image path or URL")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def resolve_source(source: str, out_dir: Path) -> tuple[Path, str]:
    """Return (local_image_path, stem_for_output_names). Download URL into out_dir if needed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if source.startswith("http://") or source.startswith("https://"):
        name = Path(urlparse(source).path).name or "sample.jpg"
        local = out_dir / name
        if not local.is_file():
            print(f"downloading {source} -> {local}")
            urllib.request.urlretrieve(source, local)
        else:
            print(f"using cached source {local}")
        return local, local.stem
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"source image not found: {path}")
    return path, path.stem


def main() -> int:
    args = parse_args()
    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} device={args.device}")
    import ultralytics

    print(f"ultralytics={ultralytics.__version__}")

    try:
        src_path, stem = resolve_source(args.source, WEIGHTS_DIR)
    except Exception as e:
        print(f"FAIL resolve source: {e}")
        return 1
    print(f"source={src_path}")

    failed: list[str] = []
    for size in sizes:
        name = f"yolo26{size}.pt"
        local_w = WEIGHTS_DIR / name
        model_path = str(local_w) if local_w.is_file() else name
        print(f"\n=== {name} ({model_path}) ===")
        try:
            t0 = time.perf_counter()
            model = YOLO(model_path)
            model.predict(str(src_path), imgsz=args.imgsz, device=args.device, verbose=False)
            t1 = time.perf_counter()
            results = model.predict(str(src_path), imgsz=args.imgsz, device=args.device, verbose=False)
            t2 = time.perf_counter()
            n_det = 0 if not results else len(results[0].boxes)
            print(
                f"OK load+warmup={t1 - t0:.2f}s infer={t2 - t1:.2f}s detections={n_det}"
            )
            if results:
                out = WEIGHTS_DIR / f"smoke_{stem}_yolo26{size}.jpg"
                results[0].save(filename=str(out))
                print(f"saved {out}")
        except Exception as e:
            failed.append(name)
            print(f"FAIL {name}: {e}")

    if failed:
        print("\nFAILED:", ", ".join(failed))
        return 1
    print("\nAll requested YOLO26 sizes passed.")
    print(f"outputs under: {WEIGHTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
