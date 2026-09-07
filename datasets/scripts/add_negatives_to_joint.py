#!/usr/bin/env python3
"""向联合 YOLO 检测集注入空标签负样本（对全部类生效）。"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_SPLIT_RATIOS = (0.7, 0.2, 0.1)


def _progress_iter(items: list[Any], *, desc: str, every: int = 2000):
    """带进度的迭代：优先 tqdm；否则周期性 print。"""
    total = len(items)
    try:
        from tqdm import tqdm  # type: ignore

        yield from tqdm(
            items,
            total=total,
            desc=desc,
            unit="img",
            mininterval=1.0,
            dynamic_ncols=True,
        )
        return
    except Exception:
        pass
    for i, item in enumerate(items, 1):
        if every > 0 and (i % every == 0 or i == total or i == 1):
            pct = int(100 * i / total) if total else 100
            print(f"[neg]   {desc}: {i}/{total} ({pct}%)", flush=True)
        yield item


def is_empty_label(label_path: Path) -> bool:
    if not label_path.is_file():
        return True
    text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    return text == ""


def iter_yolo_splits(root: Path) -> list[tuple[str, Path, Path]]:
    splits: list[tuple[str, Path, Path]] = []
    for split in ("train", "val", "test"):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        if img_dir.is_dir():
            splits.append((split, img_dir, lbl_dir))
    if splits:
        return splits
    img_dir = root / "images"
    lbl_dir = root / "labels"
    if img_dir.is_dir():
        return [("all", img_dir, lbl_dir)]
    return []


def collect_empty_from_yolo(root: Path) -> list[Path]:
    result: list[Path] = []
    for split, img_dir, lbl_dir in iter_yolo_splits(root):
        imgs = [
            p
            for p in img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        ]
        print(
            f"[neg] scan empty labels root={root} split={split} n={len(imgs)}",
            flush=True,
        )
        for img_path in _progress_iter(
            imgs, desc=f"empty {root.name}/{split}"
        ):
            label_path = lbl_dir / f"{img_path.stem}.txt"
            if is_empty_label(label_path):
                result.append(img_path.resolve())
    return result


def collect_images_recursive(dir_path: Path) -> list[Path]:
    print(f"[neg] listing images under {dir_path} ...", flush=True)
    files = [p for p in dir_path.rglob("*") if p.is_file()]
    result: list[Path] = []
    for p in _progress_iter(files, desc=f"from-images {dir_path.name}"):
        if p.suffix.lower() in IMG_EXTS:
            result.append(p.resolve())
    return result


def count_positive_images(joint_root: Path) -> int:
    count = 0
    for split, img_dir, lbl_dir in iter_yolo_splits(joint_root):
        imgs = [
            p
            for p in img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        ]
        print(
            f"[neg] count positives split={split} n={len(imgs)}",
            flush=True,
        )
        for img_path in _progress_iter(imgs, desc=f"positives {split}"):
            label_path = lbl_dir / f"{img_path.stem}.txt"
            if not is_empty_label(label_path):
                count += 1
    return count


def compute_target_count(
    *,
    candidates: int,
    positives: int,
    ratio: float | None,
    max_n: int | None,
) -> int:
    t = int(candidates)
    if ratio is not None:
        t = min(t, int(math.ceil(positives * float(ratio))))
    if max_n is not None:
        t = min(t, int(max_n))
    return max(0, t)


def sample_paths(paths: list[Path], k: int, seed: int) -> list[Path]:
    rng = random.Random(seed)
    order = list(paths)
    rng.shuffle(order)
    if k >= len(order):
        return order
    return order[:k]


def normalize_split_ratios(ratios: list[float] | None) -> tuple[float, float, float]:
    if ratios is None:
        return DEFAULT_SPLIT_RATIOS
    vals = [float(x) for x in ratios]
    if len(vals) != 3 or any(v < 0 for v in vals) or sum(vals) <= 0:
        raise ValueError(f"split_ratios must be 3 non-negative numbers summing > 0, got {ratios!r}")
    s = sum(vals)
    return (vals[0] / s, vals[1] / s, vals[2] / s)


def split_pool(
    items: list[Path],
    *,
    ratios: list[float] | None,
    seed: int,
) -> dict[str, list[Path]]:
    r_train, r_val, r_test = normalize_split_ratios(ratios)
    if not items:
        return {"train": [], "val": [], "test": []}
    rng = random.Random(seed)
    order = list(items)
    rng.shuffle(order)
    n = len(order)
    if n == 1:
        return {"train": order, "val": [], "test": []}
    if n == 2:
        return {"train": [order[0]], "val": [order[1]], "test": []}
    n_test = int(round(n * r_test))
    n_val = int(round(n * r_val))
    n_test = max(1, n_test) if r_test > 0 else 0
    n_val = max(1, n_val) if r_val > 0 else 0
    if n_test + n_val >= n:
        overflow = n_test + n_val - (n - 1)
        if n_test >= n_val and n_test > 0:
            n_test = max(0, n_test - overflow)
        else:
            n_val = max(0, n_val - overflow)
        if n_test + n_val >= n:
            n_test = max(0, min(n_test, 1))
            n_val = max(0, min(n_val, n - 1 - n_test))
    test = order[:n_test]
    val = order[n_test : n_test + n_val]
    train = order[n_test + n_val :]
    return {"train": train, "val": val, "test": test}


def _link_or_copy(src: Path, dst: Path, copy_images: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_images:
        shutil.copy2(src, dst)
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def write_neg_staging(
    *,
    staging: Path,
    by_split: dict[str, list[Path]],
    names: list[str],
    copy_images: bool = False,
) -> dict[str, Any]:
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    images_written: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    idx = 0
    for split in ("train", "val", "test"):
        srcs = list(by_split.get(split, []))
        if srcs:
            print(f"[neg] write staging split={split} n={len(srcs)}", flush=True)
        for src in _progress_iter(srcs, desc=f"staging {split}"):
            unique = f"{idx:06d}_{src.stem}{src.suffix}"
            img_dst = staging / "images" / split / unique
            lbl_dst = staging / "labels" / split / f"{Path(unique).stem}.txt"
            _link_or_copy(src, img_dst, copy_images)
            lbl_dst.parent.mkdir(parents=True, exist_ok=True)
            lbl_dst.write_text("", encoding="utf-8")
            images_written[split] += 1
            idx += 1

    cfg: dict[str, Any] = {
        "path": str(staging.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": list(names),
    }
    (staging / "data.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return {
        "staging": str(staging.resolve()),
        "images_written": images_written,
        "total": sum(images_written.values()),
    }


def load_joint_names(out_root: Path) -> list[str]:
    cfg = yaml.safe_load((out_root / "data.yaml").read_text(encoding="utf-8"))
    names = cfg.get("names") if isinstance(cfg, dict) else None
    if isinstance(names, list):
        return [str(name) for name in names]
    if isinstance(names, dict):
        indexed = {int(k): str(v) for k, v in names.items()}
        return [indexed[i] for i in sorted(indexed)]
    raise ValueError(f"bad names in {out_root / 'data.yaml'}")


def prefix_present(out_root: Path, prefix: str) -> bool:
    meta_path = out_root / "merge_meta.json"
    if not meta_path.is_file():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if isinstance(meta, dict) and meta.get("prefix") == prefix:
        return True
    sets = meta.get("sets", []) if isinstance(meta, dict) else []
    return any(
        isinstance(item, dict) and item.get("prefix") == prefix
        for item in sets
    )


def build_append_yaml(
    *, names: list[str], staging: Path, prefix: str
) -> dict[str, Any]:
    return {
        "version": 1,
        "names": list(names),
        "sets": [{"root": str(staging), "prefix": prefix}],
    }


def _validation_error(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def paths_overlap(first: Path, second: Path) -> bool:
    return (
        first == second
        or first in second.parents
        or second in first.parents
    )


def merge_script_path() -> Path:
    return ROOT / "datasets" / "scripts" / "merge_yolo_sets.py"


def remove_prefix_from_joint(out: Path, prefix: str) -> None:
    subprocess.run(
        [
            sys.executable,
            str(merge_script_path()),
            "--mode",
            "remove",
            "--out",
            str(out),
            "--prefix",
            prefix,
        ],
        check=True,
    )


def resolve_out_and_data_root(
    args: argparse.Namespace,
) -> tuple[Path, Path | None] | int:
    data_root = args.data_root
    if data_root is None:
        env_root = os.environ.get("DATA_ROOT")
        data_root = Path(env_root).expanduser() if env_root else None

    out = args.out
    if out is None:
        if data_root is None:
            return _validation_error("--out is required when DATA_ROOT is not set")
        out = data_root / "datasets" / "detect_w1_fire"
    return out.expanduser().resolve(), data_root


def run_add_negatives(args: argparse.Namespace) -> int:
    resolved = resolve_out_and_data_root(args)
    if isinstance(resolved, int):
        return resolved
    out, data_root = resolved

    if args.remove:
        if args.from_yolo or args.from_images or args.replace:
            return _validation_error(
                "--remove cannot be combined with --from-yolo/--from-images/--replace"
            )
        if not prefix_present(out, args.prefix):
            print(
                json.dumps(
                    {
                        "action": "remove",
                        "out": str(out),
                        "prefix": args.prefix,
                        "present": False,
                        "removed": False,
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "action": "remove",
                        "out": str(out),
                        "prefix": args.prefix,
                        "present": True,
                        "removed": False,
                        "dry_run": True,
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        print(
            f"[neg] removing prefix={args.prefix!r} from {out} "
            "(merge remove progress below)",
            flush=True,
        )
        remove_prefix_from_joint(out, args.prefix)
        print(
            json.dumps(
                {
                    "action": "remove",
                    "out": str(out),
                    "prefix": args.prefix,
                    "present": True,
                    "removed": True,
                },
                ensure_ascii=False,
            )
        )
        return 0

    staging = args.staging
    if staging is None:
        if data_root is not None:
            staging = (
                data_root
                / "datasets"
                / "downloads"
                / f"joint_negatives_{args.prefix}_yolo"
            )
        else:
            staging = (
                out.parent
                / "downloads"
                / f"joint_negatives_{args.prefix}_yolo"
            )
    staging = staging.expanduser().resolve()

    if paths_overlap(staging, out):
        return _validation_error(
            f"--staging and --out must not overlap: staging={staging}, out={out}"
        )

    if not args.from_yolo and not args.from_images:
        return _validation_error(
            "at least one --from-yolo or --from-images source is required"
        )

    yolo_sources = [
        source.expanduser().resolve() for source in args.from_yolo
    ]
    image_sources = [
        source.expanduser().resolve() for source in args.from_images
    ]
    for option, source in (
        *(("--from-yolo", source) for source in yolo_sources),
        *(("--from-images", source) for source in image_sources),
    ):
        if paths_overlap(staging, source):
            return _validation_error(
                f"--staging and {option} must not overlap: "
                f"staging={staging}, source={source}"
            )

    candidates: list[Path] = []
    for source in yolo_sources:
        print(f"[neg] collect empty from-yolo {source}", flush=True)
        candidates.extend(collect_empty_from_yolo(source))
    for source in image_sources:
        print(f"[neg] collect from-images {source}", flush=True)
        candidates.extend(collect_images_recursive(source))
    candidates = list(dict.fromkeys(path.resolve() for path in candidates))
    if not candidates:
        return _validation_error("no negative candidates found")

    print(f"[neg] count positives in {out} ...", flush=True)
    positives = count_positive_images(out)
    target = compute_target_count(
        candidates=len(candidates),
        positives=positives,
        ratio=args.ratio,
        max_n=args.max_n,
    )
    if target == 0:
        return _validation_error("target count is zero after applying caps")

    sampled = sample_paths(candidates, target, args.seed)
    try:
        by_split = split_pool(
            sampled, ratios=args.split_ratios, seed=args.seed
        )
    except ValueError as exc:
        return _validation_error(str(exc))

    summary: dict[str, Any] = {
        "positives": positives,
        "candidates": len(candidates),
        "target": target,
        "split_counts": {
            split: len(paths) for split, paths in by_split.items()
        },
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False))
        return 0

    present = prefix_present(out, args.prefix)
    if present and not args.replace:
        return _validation_error(
            f"prefix {args.prefix!r} already exists; use --replace"
        )

    if present:
        print(
            f"[neg] --replace: removing prefix={args.prefix!r} "
            "(merge remove progress below)",
            flush=True,
        )
        remove_prefix_from_joint(out, args.prefix)

    names = load_joint_names(out)
    print("[neg] writing staging ...", flush=True)
    staging_meta = write_neg_staging(
        staging=staging, by_split=by_split, names=names
    )
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", encoding="utf-8", delete=False
        ) as temp:
            yaml.safe_dump(
                build_append_yaml(
                    names=names, staging=staging, prefix=args.prefix
                ),
                temp,
                sort_keys=False,
                allow_unicode=True,
            )
            temp_path = Path(temp.name)
        print(
            f"[neg] merge append prefix={args.prefix!r} "
            "(merge write progress below)",
            flush=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(merge_script_path()),
                "--mode",
                "append",
                "--config",
                str(temp_path),
                "--out",
                str(out),
                "--keep-empty",
            ],
            check=True,
        )
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    summary.update(staging_meta)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--from-yolo", type=Path, action="append", default=[])
    parser.add_argument("--from-images", type=Path, action="append", default=[])
    parser.add_argument("--ratio", type=float)
    parser.add_argument("--max", dest="max_n", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        default=list(DEFAULT_SPLIT_RATIOS),
    )
    parser.add_argument("--prefix", default="bg")
    parser.add_argument("--staging", type=Path)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument(
        "--remove",
        action="store_true",
        help="只删除联合集中该 prefix 的负样本包（默认 bg），不重新注入",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--data-root", type=Path)
    return run_add_negatives(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
