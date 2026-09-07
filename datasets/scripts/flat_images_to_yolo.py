#!/usr/bin/env python3
"""把「纯某一类、无标注」的扁平图片目录转成 YOLO detect 布局。

适用：商品/白底图，整张图就是该类目标。用四角估计背景色，抠前景得到紧框。
无可用前景时回退为略内缩的整图框。

示例：
  python datasets/scripts/flat_images_to_yolo.py \\
    --src /media/.../downloads/e-bike \\
    --out /media/.../downloads/e-bike_yolo \\
    --class-name ebike
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_SPLIT_RATIOS = (0.7, 0.2, 0.1)


def _normalize_split_ratios(ratios: list[float] | tuple[float, ...] | None) -> tuple[float, float, float]:
    if ratios is None:
        return DEFAULT_SPLIT_RATIOS
    vals = [float(x) for x in ratios]
    if len(vals) != 3 or any(v < 0 for v in vals) or sum(vals) <= 0:
        raise ValueError(f"split_ratios must be 3 non-negative numbers summing > 0, got {ratios!r}")
    s = sum(vals)
    return (vals[0] / s, vals[1] / s, vals[2] / s)


def _split_pool_ratios(
    items: list[Path],
    *,
    ratios: list[float] | tuple[float, ...] | None,
    seed: int,
) -> dict[str, list[Path]]:
    r_train, r_val, r_test = _normalize_split_ratios(ratios)
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


def foreground_yolo_box(
    im: Image.Image,
    *,
    thresh: float = 28.0,
    pad_frac: float = 0.02,
    min_area_frac: float = 0.02,
) -> tuple[float, float, float, float]:
    """从近似纯色背景抠前景，返回 YOLO 归一化 (xc, yc, w, h)。"""
    rgb = np.asarray(im.convert("RGB"), dtype=np.float32)
    h, w = rgb.shape[:2]
    if h < 2 or w < 2:
        return 0.5, 0.5, 0.95, 0.95
    k = max(1, min(8, w // 20, h // 20))
    corners = np.concatenate(
        [
            rgb[:k, :k].reshape(-1, 3),
            rgb[:k, -k:].reshape(-1, 3),
            rgb[-k:, :k].reshape(-1, 3),
            rgb[-k:, -k:].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(corners, axis=0)
    diff = np.max(np.abs(rgb - bg), axis=2)
    mask = diff > thresh
    if float(mask.mean()) < min_area_frac:
        return 0.5, 0.5, 0.95, 0.95
    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    pad_x = max(1, int(round(w * pad_frac)))
    pad_y = max(1, int(round(h * pad_frac)))
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w - 1, x1 + pad_x)
    y1 = min(h - 1, y1 + pad_y)
    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    xc = (x0 + bw / 2.0) / w
    yc = (y0 + bh / 2.0) / h
    nw = bw / w
    nh = bh / h
    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    nw = min(max(nw, 1.0 / w), 1.0)
    nh = min(max(nh, 1.0 / h), 1.0)
    return float(xc), float(yc), float(nw), float(nh)


def _list_images(src: Path) -> list[Path]:
    return sorted(
        p
        for p in src.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def convert_flat_images(
    src: Path,
    out_root: Path,
    *,
    class_name: str = "ebike",
    split_ratios: tuple[float, float, float] | list[float] = DEFAULT_SPLIT_RATIOS,
    seed: int = 0,
    copy_images: bool = False,
) -> dict[str, Any]:
    src = Path(src)
    out_root = Path(out_root)
    images = _list_images(src)
    if not images:
        raise FileNotFoundError(f"no images under {src}")
    if out_root.exists():
        print(f"[flat2yolo] removing existing {out_root} ...", flush=True)
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    plan = _split_pool_ratios(images, ratios=split_ratios, seed=seed)
    images_written = {s: 0 for s in ("train", "val", "test")}
    failed = 0
    for split, imgs in plan.items():
        if not imgs:
            continue
        out_img = out_root / "images" / split
        out_lbl = out_root / "labels" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)
        n_total = len(imgs)
        print(f"[flat2yolo] split={split} n={n_total}", flush=True)
        for i, img in enumerate(imgs, 1):
            try:
                with Image.open(img) as im:
                    xc, yc, nw, nh = foreground_yolo_box(im)
            except Exception as e:
                print(f"[flat2yolo] skip {img.name}: {e}", flush=True)
                failed += 1
                continue
            stem = img.stem
            ext = img.suffix.lower()
            if ext == ".jpeg":
                ext = ".jpg"
            _link_or_copy(img, out_img / f"{stem}{ext}", copy_images)
            (out_lbl / f"{stem}.txt").write_text(
                f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n",
                encoding="utf-8",
            )
            images_written[split] += 1
            if i % 500 == 0 or i == n_total:
                print(f"[flat2yolo]   {split}: {i}/{n_total}", flush=True)

    cfg: dict[str, Any] = {
        "path": str(out_root.resolve()),
        "names": {0: class_name},
    }
    for split in ("train", "val", "test"):
        if images_written[split] > 0:
            cfg[split] = f"images/{split}"
    if "train" not in cfg and "val" in cfg:
        cfg["train"] = cfg["val"]
    if "val" not in cfg and "train" in cfg:
        cfg["val"] = cfg["train"]
    (out_root / "data.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    meta = {
        "src": str(src),
        "out": str(out_root),
        "class_name": class_name,
        "n_src": len(images),
        "failed": failed,
        "images_written": images_written,
        "split_ratios": list(_normalize_split_ratios(split_ratios)),
        "split_seed": seed,
    }
    (out_root / "convert_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True, help="扁平图片目录")
    ap.add_argument("--out", type=Path, required=True, help="输出 YOLO 根目录")
    ap.add_argument("--class-name", default="ebike")
    ap.add_argument("--split-ratios", nargs=3, type=float, default=[0.7, 0.2, 0.1])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copy-images", action="store_true")
    args = ap.parse_args(argv)
    meta = convert_flat_images(
        args.src,
        args.out,
        class_name=args.class_name,
        split_ratios=tuple(args.split_ratios),
        seed=args.seed,
        copy_images=args.copy_images,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[ok] wrote {args.out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
