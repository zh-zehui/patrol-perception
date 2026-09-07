#!/usr/bin/env python3
"""Run detect-eval for one wave: val (+test) metrics, optional baseline delta, archive.

Usage (real GPU run, from patrol_perception/, after activating pytorch_cu130):
  source /home/sribd/D1_EDU/Pytorch/switch_cuda_mode.sh 1
  cd /home/sribd/D1_EDU/patrol_perception
  PYTHONPATH=src python scripts/eval_detect_run.py \\
    --weights data/runs/detect/w1_fire/weights/best.pt \\
    --data /path/to/detect_w1_fire/data.yaml \\
    --wave w1_fire \\
    --config config/eval.yaml

Fast/smoke run on a large val set (evenly-sampled subset, no full 8k-image val):
  PYTHONPATH=src python scripts/eval_detect_run.py \\
    --weights data/weights/yolo26l.pt \\
    --data /path/to/detect_w1_fire/data.yaml \\
    --wave w1_fire_smoke_cli \\
    --max-images 400

Dry run with no GPU/ultralytics (exercises config/report/index plumbing only):
  python scripts/eval_detect_run.py \\
    --weights fake.pt --data fake_data.yaml --wave w1_fire_dry \\
    --dry-metrics-json path/to/fake_metrics.json

  fake_metrics.json shape:
    {
      "val": {"per_class": {"person": {"ap50": 0.8, "ap50_95": 0.5, ...}, ...}},
      "test": {"per_class": {...}},          # optional
      "baseline_val": {"per_class": {...}}   # optional, only used if no --baseline val run happened
    }

Note: Ultralytics' `fraction` config only subsets the *train* split (see
`ultralytics.data.build.build_yolo_dataset`); it is a no-op for `val`/`test` in the
installed version. Use `--max-images` instead for a real (GPU) subset eval.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from patrol_perception.eval.compare import (
    compute_delta,
    load_eval_config,
    resolve_baseline,
    summarize_metrics,
)
from patrol_perception.eval.metrics import extract_from_ultralytics
from patrol_perception.eval.report import write_eval_run

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_CONFIG = ROOT / "config" / "eval.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--weights", required=True, help="current weights (.pt) to evaluate")
    p.add_argument("--data", required=True, help="path to merged-set data.yaml")
    p.add_argument("--wave", required=True, help="wave key, e.g. w1_fire (must match eval.yaml focus_classes_by_wave)")
    p.add_argument(
        "--baseline",
        default=None,
        help="baseline weights for val delta; default: eval.yaml baseline.default",
    )
    p.add_argument(
        "--no-baseline",
        action="store_true",
        help="skip baseline val even if eval.yaml has baseline.default",
    )
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="path to config/eval.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--device", default=None, help="e.g. 0 or cpu; default: ultralytics auto-select")
    p.add_argument(
        "--splits",
        default=None,
        help="comma list overriding auto split detection, e.g. val,test (default: eval.yaml splits present in data.yaml)",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="evenly-sample N images per split for a fast/smoke real val (bypasses Ultralytics --fraction, which is a no-op on val/test)",
    )
    p.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="passthrough to Ultralytics val(fraction=...); NOTE: no-op for val/test split in this Ultralytics version, kept for forward-compat / documentation",
    )
    p.add_argument("--out-root", default=None, help="override eval.yaml output.root")
    p.add_argument("--index", default=None, help="override eval.yaml output.index")
    p.add_argument(
        "--dry-metrics-json",
        default=None,
        help="skip ultralytics/torch import entirely; load per-split metrics docs from this JSON to test orchestration without GPU",
    )
    return p.parse_args()


def _load_data_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _names_list(data_cfg: dict) -> list[str]:
    names = data_cfg["names"]
    if isinstance(names, dict):
        return [names[i] for i in sorted(names, key=int)]
    return list(names)


def _resolve_split_dir(data_cfg: dict, data_yaml_path: Path, split: str) -> Path | None:
    rel = data_cfg.get(split)
    if not rel:
        return None
    base = Path(data_cfg["path"]) if data_cfg.get("path") else data_yaml_path.parent
    rel_path = Path(rel)
    return rel_path if rel_path.is_absolute() else (base / rel_path)


def _list_images(dir_path: Path) -> list[Path]:
    return sorted(p for p in dir_path.rglob("*") if p.suffix.lower() in IMG_EXTS)


def _evenly_sample(items: list, n: int) -> list:
    """Evenly-spaced sample across the sorted list (keeps proportional class-source mix)."""
    if n <= 0 or n >= len(items):
        return items
    step = len(items) / n
    idxs = sorted({int(i * step) for i in range(n)})
    return [items[i] for i in idxs]


def _build_subset_data_yaml(
    data_cfg: dict, data_yaml_path: Path, splits: list[str], max_images: int, tmp_dir: Path
) -> Path:
    """Write a temp data.yaml whose split(s) point to a .txt image list (evenly-sampled subset).

    Ultralytics' BaseDataset.get_img_files() accepts a .txt file of absolute image
    paths, so this yields a *real* (GPU) subset val without touching the source dataset.
    """
    subset_cfg = dict(data_cfg)
    subset_cfg.pop("path", None)
    for split in splits:
        split_dir = _resolve_split_dir(data_cfg, data_yaml_path, split)
        if split_dir is None or not split_dir.is_dir():
            continue
        images = _list_images(split_dir)
        sampled = _evenly_sample(images, max_images)
        txt_path = tmp_dir / f"{split}_subset.txt"
        txt_path.write_text("\n".join(str(p) for p in sampled) + "\n")
        subset_cfg[split] = str(txt_path)
    out_yaml = tmp_dir / "data_subset.yaml"
    out_yaml.write_text(yaml.safe_dump(subset_cfg, allow_unicode=True, sort_keys=False))
    return out_yaml


def _run_val(
    weights: str,
    data_yaml: Path,
    split: str,
    names: list[str],
    *,
    imgsz: int,
    device: str | None,
    batch: int | None,
    fraction: float | None,
    project: Path,
    name: str,
) -> dict:
    from ultralytics import YOLO

    model = YOLO(weights)
    val_kwargs: dict = dict(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        verbose=False,
        plots=False,
        project=str(project),
        name=name,
        exist_ok=True,
    )
    if device is not None:
        val_kwargs["device"] = device
    if batch is not None:
        val_kwargs["batch"] = batch
    if fraction is not None:
        val_kwargs["fraction"] = fraction
    results = model.val(**val_kwargs)
    return extract_from_ultralytics(results, names, split=split)


def _load_dry_metrics(payload: dict, key: str, *, split: str, focus: list[str], base: list[str]) -> dict | None:
    doc = payload.get(key)
    if doc is None:
        return None
    per_class = doc.get("per_class", doc)
    return {
        "split": doc.get("split", split),
        "names": doc.get("names", list(per_class)),
        "per_class": per_class,
        "summary": summarize_metrics(per_class, focus=focus, base=base),
    }


def main() -> int:
    args = parse_args()

    config_path = Path(args.config)
    cfg = load_eval_config(config_path)
    focus_classes_by_wave = cfg.get("focus_classes_by_wave", {})
    wave_known = args.wave in focus_classes_by_wave
    if not wave_known:
        print(
            f"WARNING: wave '{args.wave}' not found in focus_classes_by_wave "
            f"({config_path}); focus classes will be empty for this run.",
            file=sys.stderr,
        )
    focus_classes = list(focus_classes_by_wave.get(args.wave, []))
    base_classes = list(cfg.get("base_classes", []))
    args.baseline = resolve_baseline(args.baseline, cfg, no_baseline=args.no_baseline)
    out_root = Path(args.out_root or cfg.get("output", {}).get("root", "data/eval_runs"))
    index_path = Path(args.index or cfg.get("output", {}).get("index", "data/eval_runs/index.jsonl"))

    data_yaml_path = Path(args.data)
    data_cfg = _load_data_yaml(data_yaml_path)
    names = _names_list(data_cfg)

    if args.splits:
        splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    else:
        cfg_splits = cfg.get("splits", ["val", "test"])
        splits = [s for s in cfg_splits if data_cfg.get(s)]
    if "val" not in splits:
        if not data_cfg.get("val"):
            print("ERROR: data.yaml has no 'val' split (required)", file=sys.stderr)
            return 2
        splits = ["val"] + splits

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / f"{args.wave}_{timestamp}"

    metrics_by_split: dict[str, dict] = {}
    baseline_doc: dict | None = None
    delta: dict | None = None
    notes: list[str] = []

    if args.dry_metrics_json:
        payload = json.loads(Path(args.dry_metrics_json).read_text())
        for split in splits:
            doc = _load_dry_metrics(payload, split, split=split, focus=focus_classes, base=base_classes)
            if doc is not None:
                metrics_by_split[split] = doc
        baseline_doc = _load_dry_metrics(
            payload, "baseline_val", split="val", focus=focus_classes, base=base_classes
        )
        notes.append("dry-metrics-json used: no GPU/ultralytics import, orchestration-only check")
    else:
        eval_data_yaml = data_yaml_path
        tmp_ctx = None
        if args.max_images:
            out_dir.mkdir(parents=True, exist_ok=True)
            tmp_ctx = tempfile.TemporaryDirectory(prefix="eval_subset_", dir=str(out_dir))
            eval_data_yaml = _build_subset_data_yaml(
                data_cfg, data_yaml_path, splits, args.max_images, Path(tmp_ctx.name)
            )
            notes.append(
                f"subset eval: --max-images={args.max_images} per split, evenly sampled "
                "(Ultralytics --fraction is a no-op on val/test, so a temp image-list data.yaml was used instead)"
            )
        try:
            for split in splits:
                doc = _run_val(
                    args.weights,
                    eval_data_yaml,
                    split,
                    names,
                    imgsz=args.imgsz,
                    device=args.device,
                    batch=args.batch,
                    fraction=args.fraction,
                    project=out_dir / "ultralytics_raw",
                    name=split,
                )
                doc["summary"] = summarize_metrics(doc["per_class"], focus=focus_classes, base=base_classes)
                metrics_by_split[split] = doc

            if args.baseline:
                baseline_doc = _run_val(
                    args.baseline,
                    eval_data_yaml,
                    "val",
                    names,
                    imgsz=args.imgsz,
                    device=args.device,
                    batch=args.batch,
                    fraction=args.fraction,
                    project=out_dir / "ultralytics_raw",
                    name="baseline_val",
                )
                baseline_doc["summary"] = summarize_metrics(
                    baseline_doc["per_class"], focus=focus_classes, base=base_classes
                )
        finally:
            if tmp_ctx is not None:
                tmp_ctx.cleanup()

    if baseline_doc is not None and "val" in metrics_by_split:
        delta = compute_delta(metrics_by_split["val"], baseline_doc, base_classes=base_classes)

    meta = {
        "wave": args.wave,
        "weights": args.weights,
        "data": str(data_yaml_path),
        "baseline": args.baseline,
        "imgsz": args.imgsz,
        "splits": splits,
        "names": names,
        "focus_classes": focus_classes,
        "base_classes": base_classes,
        "wave_known": wave_known,
        "eval_config": str(config_path),
        "ts": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
    }

    write_eval_run(
        out_dir,
        meta=meta,
        metrics_by_split=metrics_by_split,
        baseline_val=baseline_doc,
        delta=delta,
        index_path=index_path,
    )

    val_summary = metrics_by_split.get("val", {}).get("summary", {})
    print(f"eval run written to: {out_dir.resolve()}")
    print(
        "val: map50={map50} focus_map50={focus_map50} base_map50={base_map50}".format(
            map50=val_summary.get("map50"),
            focus_map50=val_summary.get("focus_map50"),
            base_map50=val_summary.get("base_map50"),
        )
    )
    if delta is not None:
        print(f"delta_base_map50={delta.get('delta_base_map50')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
