#!/usr/bin/env python3
"""TACO COCO JSON → YOLO detect 布局（按 class_maps/taco_litter.yaml 映射/过滤）。"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAP = ROOT / "config" / "class_maps" / "taco_litter.yaml"


def load_class_map(path: Path) -> tuple[list[str], dict[str, str]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = list(data["target_names"])
    raw_map = dict(data.get("map") or {})
    unknown = sorted(set(raw_map.values()) - set(names))
    if unknown:
        raise ValueError(f"map targets not in target_names: {unknown}")
    return names, raw_map


def coco_bbox_to_yolo(
    bbox: list[float],
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float] | None:
    """COCO [x,y,w,h] 像素 → YOLO 归一化 xc yc w h；无效框返回 None。"""
    if img_w <= 0 or img_h <= 0 or len(bbox) < 4:
        return None
    x, y, w, h = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    if w <= 0 or h <= 0:
        return None
    xc = (x + w / 2.0) / img_w
    yc = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    # 轻微越界裁剪
    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    nw = min(max(nw, 0.0), 1.0)
    nh = min(max(nh, 0.0), 1.0)
    if nw <= 0 or nh <= 0:
        return None
    return xc, yc, nw, nh


def build_name_to_id(categories: list[dict]) -> dict[int, str]:
    return {int(c["id"]): str(c["name"]) for c in categories}


def convert_taco(
    taco_root: Path,
    out_root: Path,
    class_map_path: Path,
    *,
    val_ratio: float = 0.15,
    seed: int = 42,
    copy_images: bool = False,
    keep_empty: bool = False,
    annotations_name: str = "annotations.json",
) -> dict[str, Any]:
    """
    写出：
      out_root/images/{train,val}/...
      out_root/labels/{train,val}/...
      out_root/data.yaml
    仅保留映射表中的类；无保留框的图默认跳过（keep_empty=True 则写空 txt）。
    """
    target_names, src_to_tgt = load_class_map(class_map_path)
    tgt_to_id = {n: i for i, n in enumerate(target_names)}

    data_dir = taco_root / "data"
    ann_path = data_dir / annotations_name
    if not ann_path.is_file():
        raise FileNotFoundError(f"missing {ann_path}")

    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    id_to_src_name = build_name_to_id(coco.get("categories") or [])
    images = {int(im["id"]): im for im in (coco.get("images") or [])}

    # image_id -> list of yolo lines
    labels: dict[int, list[str]] = defaultdict(list)
    dropped_ann = 0
    kept_ann = 0
    for ann in coco.get("annotations") or []:
        if ann.get("iscrowd"):
            dropped_ann += 1
            continue
        src_name = id_to_src_name.get(int(ann["category_id"]))
        if not src_name or src_name not in src_to_tgt:
            dropped_ann += 1
            continue
        im = images.get(int(ann["image_id"]))
        if not im:
            dropped_ann += 1
            continue
        yolo = coco_bbox_to_yolo(
            ann.get("bbox") or [],
            int(im["width"]),
            int(im["height"]),
        )
        if yolo is None:
            dropped_ann += 1
            continue
        cls_id = tgt_to_id[src_to_tgt[src_name]]
        xc, yc, nw, nh = yolo
        labels[int(ann["image_id"])].append(
            f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}"
        )
        kept_ann += 1

    # 候选图：有标签，或 keep_empty 时全部有文件的图
    candidates: list[int] = []
    missing_img = 0
    for iid, im in images.items():
        src_file = data_dir / im["file_name"]
        if not src_file.is_file():
            missing_img += 1
            continue
        if labels[iid] or keep_empty:
            candidates.append(iid)

    rng = random.Random(seed)
    rng.shuffle(candidates)
    n_val = int(round(len(candidates) * val_ratio)) if candidates else 0
    val_ids = set(candidates[:n_val])

    for split in ("train", "val"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    def _link_or_copy(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if copy_images:
            shutil.copy2(src, dst)
        else:
            try:
                dst.symlink_to(src.resolve())
            except OSError:
                shutil.copy2(src, dst)

    written = {"train": 0, "val": 0}
    for iid in candidates:
        im = images[iid]
        split = "val" if iid in val_ids else "train"
        src = data_dir / im["file_name"]
        # 扁平化文件名，避免子目录
        flat = im["file_name"].replace("/", "__")
        dst_img = out_root / "images" / split / flat
        dst_lbl = out_root / "labels" / split / (Path(flat).stem + ".txt")
        _link_or_copy(src, dst_img)
        lines = labels.get(iid) or []
        dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written[split] += 1

    data_yaml = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(target_names)},
    }
    (out_root / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    meta = {
        "taco_root": str(taco_root),
        "out_root": str(out_root),
        "class_map": str(class_map_path),
        "kept_annotations": kept_ann,
        "dropped_annotations": dropped_ann,
        "missing_images": missing_img,
        "images_train": written["train"],
        "images_val": written["val"],
        "target_names": target_names,
    }
    (out_root / "convert_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--taco-root",
        type=Path,
        required=True,
        help="TACO 根目录（含 data/annotations.json 与 batch_*）",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="输出 YOLO 根目录",
    )
    ap.add_argument(
        "--map",
        type=Path,
        default=DEFAULT_MAP,
        help="类映射 yaml",
    )
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--copy-images",
        action="store_true",
        help="复制图片；默认尝试 symlink",
    )
    ap.add_argument(
        "--keep-empty",
        action="store_true",
        help="无目标图也保留（空 labels），作负样本",
    )
    ap.add_argument(
        "--annotations",
        type=str,
        default="annotations.json",
        help="data/ 下标注文件名",
    )
    args = ap.parse_args(argv)

    meta = convert_taco(
        args.taco_root,
        args.out,
        args.map,
        val_ratio=args.val_ratio,
        seed=args.seed,
        copy_images=args.copy_images,
        keep_empty=args.keep_empty,
        annotations_name=args.annotations,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[ok] wrote {args.out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
