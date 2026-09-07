#!/usr/bin/env python3
"""已有 YOLO detect 集：按 class_map 过滤/重映射类 id，写出新 data.yaml。"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_remap_map(path: Path) -> tuple[list[str], dict[str, str]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = list(data["target_names"])
    raw = {str(k): str(v) for k, v in (data.get("map") or {}).items()}
    # 大小写不敏感匹配时用 lower 键再查
    unknown = sorted(set(raw.values()) - set(names))
    if unknown:
        raise ValueError(f"map targets not in target_names: {unknown}")
    return names, raw


def load_source_names(data_yaml: Path) -> dict[int, str]:
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = cfg.get("names")
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    raise ValueError(f"bad names in {data_yaml}")


def ensure_data_yaml(
    src_root: Path,
    *,
    names: dict[int, str] | None = None,
) -> Path:
    """若缺少 data.yaml，按 images/{train,val,test} 布局补写。"""
    data_yaml = src_root / "data.yaml"
    if data_yaml.is_file():
        return data_yaml
    if not names:
        raise FileNotFoundError(
            f"missing {data_yaml}；请补 data.yaml，或传 --names 'ebike,car,person,bicycle,sign'"
        )
    cfg: dict[str, Any] = {
        "path": str(src_root.resolve()),
        "names": {int(k): str(v) for k, v in sorted(names.items())},
    }
    for split in ("train", "val", "test"):
        if (src_root / "images" / split).is_dir():
            cfg[split] = f"images/{split}"
        elif (src_root / split / "images").is_dir():
            # 仍写标准 images/split，discover 会回退
            cfg[split] = f"images/{split}"
    if "train" not in cfg and "val" in cfg:
        cfg["train"] = cfg["val"]
    data_yaml.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"[remap] wrote missing {data_yaml}")
    return data_yaml


def _lookup_target(src_name: str, src_to_tgt: dict[str, str]) -> str | None:
    if src_name in src_to_tgt:
        return src_to_tgt[src_name]
    low = {k.lower(): v for k, v in src_to_tgt.items()}
    return low.get(src_name.lower())


def remap_label_lines(
    lines: list[str],
    src_id_to_name: dict[int, str],
    src_to_tgt: dict[str, str],
    tgt_to_id: dict[str, int],
) -> tuple[list[str], int, int]:
    """返回 (新行, kept, dropped)。"""
    out: list[str] = []
    kept = dropped = 0
    for line in lines:
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 5:
            dropped += 1
            continue
        old_id = int(float(parts[0]))
        src_name = src_id_to_name.get(old_id)
        tgt = _lookup_target(src_name, src_to_tgt) if src_name else None
        if not tgt:
            dropped += 1
            continue
        new_id = tgt_to_id[tgt]
        out.append(" ".join([str(new_id), *parts[1:]]))
        kept += 1
    return out, kept, dropped


def _link_or_copy(src: Path, dst: Path, copy_images: bool) -> None:
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


def _labels_dir_for_images(img_dir: Path) -> Path:
    # .../train/images → .../train/labels  或  .../images/train → .../labels/train
    if img_dir.name == "images":
        return img_dir.parent / "labels"
    if img_dir.parent.name == "images":
        return img_dir.parent.parent / "labels" / img_dir.name
    return img_dir.parent / "labels"


def discover_splits(src_root: Path, data_yaml: Path) -> dict[str, Path]:
    """
    返回 {规范split名: images目录}。
    规范名: train / val / test
    支持:
      - images/train
      - train/images、valid/images
      - data.yaml 里的相对路径（错误的 ../train 会回退探测）
    """
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    alias = {"train": "train", "val": "val", "valid": "val", "test": "test"}
    found: dict[str, Path] = {}

    for key in ("train", "val", "valid", "test"):
        rel = cfg.get(key)
        if not rel:
            continue
        cand = (data_yaml.parent / str(rel)).resolve()
        if cand.is_dir():
            found[alias[key]] = cand

    # 回退探测
    probes = [
        ("train", src_root / "images" / "train"),
        ("val", src_root / "images" / "val"),
        ("test", src_root / "images" / "test"),
        ("train", src_root / "train" / "images"),
        ("val", src_root / "valid" / "images"),
        ("val", src_root / "val" / "images"),
        ("test", src_root / "test" / "images"),
    ]
    for split, path in probes:
        if split not in found and path.is_dir():
            found[split] = path.resolve()

    return found


def remap_yolo_set(
    src_root: Path,
    out_root: Path,
    class_map_path: Path,
    *,
    copy_images: bool = False,
    keep_empty: bool = False,
    source_names: dict[int, str] | None = None,
) -> dict[str, Any]:
    data_yaml = ensure_data_yaml(src_root, names=source_names)

    target_names, src_to_tgt = load_remap_map(class_map_path)
    tgt_to_id = {n: i for i, n in enumerate(target_names)}
    src_id_to_name = load_source_names(data_yaml)
    splits = discover_splits(src_root, data_yaml)
    if not splits:
        raise FileNotFoundError(
            f"在 {src_root} 未找到 images 目录（尝试了 images/train 与 train/images）"
        )

    kept_boxes = dropped_boxes = 0
    images_written = {s: 0 for s in ("train", "val", "test")}
    images_skipped_empty = 0

    for split, img_dir in splits.items():
        lbl_dir = _labels_dir_for_images(img_dir)
        out_img = out_root / "images" / split
        out_lbl = out_root / "labels" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        for img in sorted(img_dir.iterdir()):
            if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            raw_lines = (
                lbl.read_text(encoding="utf-8").splitlines() if lbl.is_file() else []
            )
            new_lines, k, d = remap_label_lines(
                raw_lines, src_id_to_name, src_to_tgt, tgt_to_id
            )
            kept_boxes += k
            dropped_boxes += d
            if not new_lines and not keep_empty:
                images_skipped_empty += 1
                continue
            _link_or_copy(img, out_img / img.name, copy_images)
            (out_lbl / (img.stem + ".txt")).write_text(
                ("\n".join(new_lines) + ("\n" if new_lines else "")),
                encoding="utf-8",
            )
            images_written[split] += 1

    out_cfg: dict[str, Any] = {
        "path": str(out_root.resolve()),
        "names": {i: n for i, n in enumerate(target_names)},
    }
    for split in ("train", "val", "test"):
        if images_written.get(split, 0) > 0:
            out_cfg[split] = f"images/{split}"
    if "train" not in out_cfg and images_written.get("val", 0) > 0:
        out_cfg["train"] = "images/val"
    (out_root / "data.yaml").write_text(
        yaml.safe_dump(out_cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    meta = {
        "src": str(src_root),
        "out": str(out_root),
        "class_map": str(class_map_path),
        "kept_boxes": kept_boxes,
        "dropped_boxes": dropped_boxes,
        "images_written": images_written,
        "images_skipped_empty": images_skipped_empty,
        "target_names": target_names,
        "source_names": src_id_to_name,
        "discovered_splits": {k: str(v) for k, v in splits.items()},
    }
    (out_root / "remap_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True, help="源 YOLO 集根目录（含 data.yaml）")
    ap.add_argument("--out", type=Path, required=True, help="输出目录")
    ap.add_argument(
        "--map",
        type=Path,
        default=ROOT / "config/class_maps/cigarette_face.yaml",
    )
    ap.add_argument("--copy-images", action="store_true")
    ap.add_argument(
        "--keep-empty",
        action="store_true",
        help="过滤后无框的图也保留（空 txt）",
    )
    ap.add_argument(
        "--names",
        default="",
        help="源集缺 data.yaml 时补写类名，逗号分隔，如 ebike,car,person,bicycle,sign",
    )
    args = ap.parse_args(argv)
    names = None
    if args.names.strip():
        names = {i: n.strip() for i, n in enumerate(args.names.split(",")) if n.strip()}
    meta = remap_yolo_set(
        args.src,
        args.out,
        args.map,
        copy_images=args.copy_images,
        keep_empty=args.keep_empty,
        source_names=names,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[ok] wrote {args.out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
