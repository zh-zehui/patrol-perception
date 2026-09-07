#!/usr/bin/env python3
"""统计 YOLO 检测集各类 box 数 / 含该类的图数。

适用于 downloads/ 下的公开集、remap 产物、以及 merge 后的 detect_w1_fire 等。

示例::

  cd /home/sribd/D1_EDU/patrol_perception
  python datasets/scripts/summarize_yolo_dataset.py \\
    --root /media/sribd/.../datasets/downloads/coco_val2017 \\
    --classes person,bicycle,car,truck

  python datasets/scripts/summarize_yolo_dataset.py \\
    --root /media/sribd/.../datasets/detect_w1_fire
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


def _progress_iter(items: list[Any], *, desc: str, every: int = 2000):
    """带进度的迭代：优先 tqdm；否则周期性 print。"""
    total = len(items)
    try:
        from tqdm import tqdm  # type: ignore

        yield from tqdm(
            items,
            total=total,
            desc=desc,
            unit="lbl",
            mininterval=1.0,
            dynamic_ncols=True,
        )
        return
    except Exception:
        pass
    for i, item in enumerate(items, 1):
        if every > 0 and (i % every == 0 or i == total or i == 1):
            pct = int(100 * i / total) if total else 100
            print(f"[summarize]   {desc}: {i}/{total} ({pct}%)", flush=True)
        yield item


# COCO 80 名（仅当 data.yaml 缺失且未传 --names 时作后备）
_COCO80 = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def load_names(
    root: Path,
    *,
    names_override: dict[int, str] | None = None,
) -> dict[int, str]:
    if names_override:
        return {int(k): str(v) for k, v in names_override.items()}
    data_yaml = root / "data.yaml"
    if data_yaml.is_file():
        cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
        names = cfg.get("names")
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, list):
            return {i: str(n) for i, n in enumerate(names)}
    return {i: n for i, n in enumerate(_COCO80)}


def discover_label_dirs(root: Path) -> list[tuple[str, Path]]:
    """返回 (split_name, labels_dir)。支持 labels/{train,val,test} 与 COCO 式 labels/val2017。"""
    labels_root = root / "labels"
    if not labels_root.is_dir():
        raise FileNotFoundError(f"missing labels/ under {root}")

    found: list[tuple[str, Path]] = []
    for child in sorted(labels_root.iterdir()):
        if child.is_dir() and any(child.glob("*.txt")):
            found.append((child.name, child))
    # 少数布局把 txt 直接放在 labels/
    if not found and any(labels_root.glob("*.txt")):
        found.append(("labels", labels_root))
    if not found:
        raise FileNotFoundError(f"no *.txt under {labels_root}")
    return found


def _scan_split(
    label_dir: Path,
    *,
    id_to_name: dict[int, str],
    class_ids: set[int] | None,
) -> dict[str, Any]:
    boxes: Counter[int] = Counter()
    imgs: Counter[int] = Counter()
    n_files = 0
    empty = 0
    skipped_ids: Counter[int] = Counter()

    files = sorted(label_dir.glob("*.txt"))
    print(
        f"[summarize] scan labels split={label_dir.name} n={len(files)}",
        flush=True,
    )
    for p in _progress_iter(files, desc=f"labels {label_dir.name}"):
        n_files += 1
        lines = [ln.strip() for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
        if not lines:
            empty += 1
            continue
        seen: set[int] = set()
        for ln in lines:
            parts = ln.split()
            if not parts:
                continue
            try:
                cid = int(float(parts[0]))
            except ValueError:
                continue
            if class_ids is not None and cid not in class_ids:
                skipped_ids[cid] += 1
                continue
            boxes[cid] += 1
            seen.add(cid)
        for c in seen:
            imgs[c] += 1

    total_boxes = int(sum(boxes.values()))
    boxes_by_class: dict[str, int] = {}
    images_by_class: dict[str, int] = {}
    # 稳定顺序：先按 id_to_name 键序，再补未知 id
    ordered_ids = sorted(id_to_name.keys())
    for cid in ordered_ids:
        if class_ids is not None and cid not in class_ids:
            continue
        name = id_to_name.get(cid, str(cid))
        boxes_by_class[name] = int(boxes.get(cid, 0))
        images_by_class[name] = int(imgs.get(cid, 0))
    for cid in sorted(set(boxes) - set(ordered_ids)):
        name = id_to_name.get(cid, str(cid))
        boxes_by_class[name] = int(boxes[cid])
        images_by_class[name] = int(imgs[cid])

    return {
        "label_dir": str(label_dir),
        "label_files": n_files,
        "empty_labels": empty,
        "total_boxes": total_boxes,
        "boxes_by_class": boxes_by_class,
        "images_by_class": images_by_class,
        "skipped_boxes_by_id": {str(k): int(v) for k, v in sorted(skipped_ids.items())},
    }


def summarize_dataset(
    root: Path,
    *,
    class_ids: set[int] | None = None,
    names_override: dict[int, str] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    id_to_name = load_names(root, names_override=names_override)
    names_list = [id_to_name[i] for i in sorted(id_to_name)]
    if class_ids is not None:
        names_list = [id_to_name[i] for i in sorted(class_ids) if i in id_to_name]
        # 允许 filter 里有 yaml 未列的 id（如 COCO truck=7）
        for i in sorted(class_ids):
            if i not in id_to_name:
                id_to_name[i] = str(i)
                names_list.append(str(i))

    splits: dict[str, Any] = {}
    for split_name, label_dir in discover_label_dirs(root):
        # 若传了 names_override 且含稀疏 id，扫描时用完整 id_to_name
        scan_names = dict(id_to_name)
        if names_override:
            scan_names.update({int(k): str(v) for k, v in names_override.items()})
        splits[split_name] = _scan_split(
            label_dir,
            id_to_name=scan_names,
            class_ids=class_ids,
        )

    return {
        "root": str(root),
        "names": names_list,
        "splits": splits,
    }


def _resolve_class_filter(
    names_csv: str | None,
    id_to_name: dict[int, str],
) -> set[int] | None:
    if not names_csv:
        return None
    wanted = {x.strip() for x in names_csv.split(",") if x.strip()}
    if not wanted:
        return None
    name_to_id = {v: k for k, v in id_to_name.items()}
    # 也允许数字 id
    out: set[int] = set()
    unknown: list[str] = []
    for w in wanted:
        if w.isdigit():
            out.add(int(w))
            continue
        if w in name_to_id:
            out.add(name_to_id[w])
            continue
        # COCO 常见别名：用户写 truck 但 yaml 可能不全
        low = {k.lower(): v for k, v in name_to_id.items()}
        if w.lower() in low:
            out.add(low[w.lower()])
            continue
        unknown.append(w)
    if unknown:
        # 对纯 COCO 后备：再试 _COCO80
        coco_map = {n: i for i, n in enumerate(_COCO80)}
        still = []
        for w in unknown:
            if w in coco_map:
                out.add(coco_map[w])
            elif w.lower().replace(" ", "_") in {n.replace(" ", "_"): i for i, n in enumerate(_COCO80)}:
                key = w.lower().replace(" ", "_")
                for i, n in enumerate(_COCO80):
                    if n.replace(" ", "_") == key:
                        out.add(i)
                        break
            else:
                still.append(w)
        if still:
            raise SystemExit(f"unknown class names: {still}; known={sorted(name_to_id)[:40]}...")
    return out


def format_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"root: {report['root']}")
    lines.append(f"names: {report['names']}")
    for split, stats in report["splits"].items():
        total = max(int(stats["total_boxes"]), 1)
        n_files = max(int(stats["label_files"]), 1)
        lines.append("")
        lines.append(
            f"=== {split}  labels={stats['label_files']}  empty={stats['empty_labels']}  "
            f"boxes={stats['total_boxes']} ==="
        )
        lines.append(f"{'name':<16} {'boxes':>8} {'%box':>7} {'imgs':>7} {'%img':>7}")
        for name, b in stats["boxes_by_class"].items():
            im = int(stats["images_by_class"].get(name, 0))
            lines.append(
                f"{name:<16} {b:8d} {100.0 * b / total:6.2f}% {im:7d} {100.0 * im / n_files:6.2f}%"
            )
        skipped = stats.get("skipped_boxes_by_id") or {}
        if skipped:
            lines.append(f"(filtered-out boxes by id: {skipped})")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True, help="数据集根目录（含 labels/，最好有 data.yaml）")
    p.add_argument(
        "--classes",
        default=None,
        help="只统计这些类，逗号分隔，如 person,bicycle,car,truck；也可写数字 id",
    )
    p.add_argument(
        "--names",
        default=None,
        help="覆盖类名，逗号分隔按 id 0..n-1；或 id:name 对如 0:person,1:bicycle,7:truck",
    )
    p.add_argument("--json", type=Path, default=None, help="可选：把完整报告写入 JSON")
    return p.parse_args()


def _parse_names_override(raw: str | None) -> dict[int, str] | None:
    if not raw:
        return None
    raw = raw.strip()
    if ":" in raw:
        out: dict[int, str] = {}
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            k, v = part.split(":", 1)
            out[int(k.strip())] = v.strip()
        return out
    return {i: n.strip() for i, n in enumerate(raw.split(",")) if n.strip()}


def main() -> None:
    args = parse_args()
    root = args.root
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    names_override = _parse_names_override(args.names)
    id_to_name = load_names(root, names_override=names_override)
    # 若用户只要 COCO 四类且 yaml 是 80 类或缺失，用 COCO 名解析 filter
    class_ids = _resolve_class_filter(args.classes, id_to_name)

    report = summarize_dataset(root, class_ids=class_ids, names_override=names_override)
    text = format_report(report)
    print(text, end="")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[wrote] {args.json}")


if __name__ == "__main__":
    main()
