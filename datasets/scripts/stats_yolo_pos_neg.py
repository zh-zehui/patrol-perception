#!/usr/bin/env python3
"""统计 YOLO 检测集：正/负样本、各类 boxes/图数、按 split 分布。

适用于已划分 images/{train,val,test} + labels/... 的布局（如实采 litter 集），
也支持扁平 images/ + labels/。

示例::

  cd /home/sribd/D1_EDU/patrol_perception
  python datasets/scripts/stats_yolo_pos_neg.py \\
    --root /media/.../datasets/downloads/02_trash_full_yolo_5271_images

  # 显示工程类名映射（仅报告，不改文件）
  python datasets/scripts/stats_yolo_pos_neg.py \\
    --root .../02_trash_full_yolo_5271_images \\
    --rename bottle:litter_bottle,box:litter_cardboard,paper:litter_paper,bag:litter_plastic_bag

  # 落盘 JSON
  python datasets/scripts/stats_yolo_pos_neg.py --root ... --json /tmp/trash_stats.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_names(root: Path) -> dict[int, str]:
    data_yaml = root / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"missing {data_yaml}")
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = cfg.get("names")
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    raise ValueError(f"bad names in {data_yaml}")


def parse_rename(spec: str) -> dict[str, str]:
    """'a:b,c:d' → {a:b, c:d}"""
    out: dict[str, str] = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"bad --rename token {part!r}, expect src:dst")
        src, dst = part.split(":", 1)
        out[src.strip()] = dst.strip()
    return out


def discover_splits(root: Path) -> list[tuple[str, Path, Path]]:
    """返回 [(split, images_dir, labels_dir), ...]。"""
    images = root / "images"
    labels = root / "labels"
    if not images.is_dir() or not labels.is_dir():
        raise FileNotFoundError(f"need {images} and {labels}")

    found: list[tuple[str, Path, Path]] = []
    for split in ("train", "val", "test"):
        idir, ldir = images / split, labels / split
        if idir.is_dir() and ldir.is_dir():
            found.append((split, idir, ldir))
    if found:
        return found

    # 扁平：images/*.jpg + labels/*.txt
    if any(p.is_file() and p.suffix.lower() in IMG_EXTS for p in images.iterdir()):
        return [("all", images, labels)]
    raise FileNotFoundError(f"no train/val/test or flat images under {root}")


def _iter_images(img_dir: Path) -> list[Path]:
    return sorted(
        p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def scan_dataset(
    root: Path,
    *,
    rename: dict[str, str] | None = None,
) -> dict[str, Any]:
    root = Path(root)
    id2n = load_names(root)
    rename = rename or {}
    splits_meta: dict[str, Any] = {}
    overall = {"pos": 0, "neg": 0, "missing_label": 0, "images": 0}
    class_boxes: Counter[str] = Counter()
    class_imgs: Counter[str] = Counter()
    class_imgs_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    class_boxes_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    multi_class_pos = 0
    boxes_on_pos: list[int] = []

    for split, idir, ldir in discover_splits(root):
        imgs = _iter_images(idir)
        sp = {"images": len(imgs), "pos": 0, "neg": 0, "missing_label": 0}
        overall["images"] += len(imgs)
        for im in imgs:
            lp = ldir / f"{im.stem}.txt"
            if not lp.is_file():
                sp["missing_label"] += 1
                overall["missing_label"] += 1
                continue
            text = lp.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                sp["neg"] += 1
                overall["neg"] += 1
                continue
            sp["pos"] += 1
            overall["pos"] += 1
            present: set[str] = set()
            nbox = 0
            for line in text.splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    cid = int(float(parts[0]))
                except ValueError:
                    continue
                src = id2n.get(cid, f"id_{cid}")
                name = rename.get(src, src)
                class_boxes[name] += 1
                class_boxes_by_split[split][name] += 1
                present.add(name)
                nbox += 1
            for c in present:
                class_imgs[c] += 1
                class_imgs_by_split[split][c] += 1
            if len(present) > 1:
                multi_class_pos += 1
            boxes_on_pos.append(nbox)
        sp["neg_pos_ratio"] = round(sp["neg"] / sp["pos"], 4) if sp["pos"] else None
        splits_meta[split] = sp

    pos = overall["pos"]
    total_boxes = int(sum(class_boxes.values()))
    # 稳定类顺序：data.yaml id 顺序（映射后）
    ordered_src = [id2n[i] for i in sorted(id2n)]
    ordered = []
    seen: set[str] = set()
    for src in ordered_src:
        dst = rename.get(src, src)
        if dst not in seen:
            ordered.append(dst)
            seen.add(dst)
    for name in sorted(class_boxes.keys()):
        if name not in seen:
            ordered.append(name)

    per_class = []
    for name in ordered:
        b = int(class_boxes.get(name, 0))
        imc = int(class_imgs.get(name, 0))
        per_class.append(
            {
                "name": name,
                "boxes": b,
                "box_pct": round(100.0 * b / total_boxes, 2) if total_boxes else 0.0,
                "images": imc,
                "pos_pct": round(100.0 * imc / pos, 2) if pos else 0.0,
                "images_by_split": {
                    s: int(class_imgs_by_split[s].get(name, 0)) for s in splits_meta
                },
                "boxes_by_split": {
                    s: int(class_boxes_by_split[s].get(name, 0)) for s in splits_meta
                },
            }
        )

    return {
        "root": str(root.resolve()),
        "names_src": {str(k): v for k, v in sorted(id2n.items())},
        "rename": rename,
        "overall": {
            **overall,
            "neg_pos_ratio": round(overall["neg"] / pos, 4) if pos else None,
            "boxes_on_pos_sum": int(sum(boxes_on_pos)),
            "boxes_on_pos_mean": round(sum(boxes_on_pos) / pos, 4) if pos else None,
            "multi_class_pos_images": multi_class_pos,
        },
        "splits": splits_meta,
        "per_class": per_class,
    }


def print_report(stats: dict[str, Any]) -> None:
    o = stats["overall"]
    print(f"root: {stats['root']}")
    print(f"names_src: {stats['names_src']}")
    if stats.get("rename"):
        print(f"rename: {stats['rename']}")
    print()
    print("=== overall ===")
    print(
        f"images={o['images']}  pos={o['pos']}  neg={o['neg']}  "
        f"missing_label={o['missing_label']}"
    )
    print(f"neg:pos = {o['neg_pos_ratio']}")
    print(
        f"boxes on pos: sum={o['boxes_on_pos_sum']} mean={o['boxes_on_pos_mean']}  "
        f"multi_class_pos={o['multi_class_pos_images']}"
    )
    print()
    print("=== splits ===")
    print(f"{'split':6} {'imgs':>7} {'pos':>7} {'neg':>7} {'miss':>5}  neg:pos")
    for split, sp in stats["splits"].items():
        print(
            f"{split:6} {sp['images']:7d} {sp['pos']:7d} {sp['neg']:7d} "
            f"{sp['missing_label']:5d}  {sp['neg_pos_ratio']}"
        )
    print()
    print("=== per-class ===")
    print(f"{'name':24} {'boxes':>7} {'%box':>7} {'imgs':>7} {'%pos':>7}")
    for row in stats["per_class"]:
        print(
            f"{row['name']:24} {row['boxes']:7d} {row['box_pct']:6.2f}% "
            f"{row['images']:7d} {row['pos_pct']:6.2f}%"
        )
    splits = list(stats["splits"].keys())
    if len(splits) > 1:
        print()
        print("=== class images by split ===")
        hdr = f"{'name':24} " + " ".join(f"{s:>7}" for s in splits)
        print(hdr)
        for row in stats["per_class"]:
            cols = " ".join(
                f"{row['images_by_split'].get(s, 0):7d}" for s in splits
            )
            print(f"{row['name']:24} {cols}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, required=True, help="YOLO 集根（含 data.yaml）")
    ap.add_argument(
        "--rename",
        default="",
        help="报告用类名映射，逗号分隔 src:dst（不改磁盘），如 bottle:litter_bottle",
    )
    ap.add_argument("--json", type=Path, default=None, help="可选，写入 JSON")
    args = ap.parse_args(argv)

    rename = parse_rename(args.rename)
    stats = scan_dataset(args.root, rename=rename)
    print_report(stats)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n[ok] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
