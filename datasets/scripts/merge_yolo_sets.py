#!/usr/bin/env python3
"""合并多个 YOLO detect 集到统一 names / 布局（如 detect_v1 或 detect_w1_fire）。

每个子集用本集 data.yaml 的 names，按 keep（类名列表）过滤并映射到全局 id。
图片默认软链，文件名加 prefix 避免碰撞。

示例：
  python datasets/scripts/merge_yolo_sets.py \\
    --config config/merge_w1_fire.yaml \\
    --out /media/.../datasets/detect_w1_fire
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_VAL_FRACTION = 0.2
DEFAULT_SPLIT_RATIOS = (0.7, 0.2, 0.1)  # train : val : test


def load_source_names(data_yaml: Path) -> dict[int, str]:
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = cfg.get("names")
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    raise ValueError(f"bad names in {data_yaml}")


def build_id_map(
    src_id_to_name: dict[int, str],
    global_names: list[str],
    keep: list[str] | None = None,
) -> dict[int, int]:
    """源类 id → 全局 id；不在 keep / 全局 names 中的丢弃。"""
    g2i = {n: i for i, n in enumerate(global_names)}
    keep_set = set(keep) if keep is not None else set(global_names)
    out: dict[int, int] = {}
    for sid, sname in src_id_to_name.items():
        if sname not in keep_set:
            continue
        if sname not in g2i:
            raise ValueError(f"kept class {sname!r} not in global names {global_names}")
        out[sid] = g2i[sname]
    return out


def remap_label_file(
    src: Path,
    dst: Path,
    id_map: dict[int, int],
) -> tuple[int, int]:
    """按 id_map 重写标签文件。返回 (kept, dropped)。"""
    kept = dropped = 0
    lines_out: list[str] = []
    if src.is_file():
        for line in src.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 5:
                dropped += 1
                continue
            old_id = int(float(parts[0]))
            if old_id not in id_map:
                dropped += 1
                continue
            lines_out.append(" ".join([str(id_map[old_id]), *parts[1:]]))
            kept += 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(
        ("\n".join(lines_out) + ("\n" if lines_out else "")),
        encoding="utf-8",
    )
    return kept, dropped


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


def _discover_splits(src_root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        p = src_root / "images" / split
        if p.is_dir():
            found[split] = p
    # COCO 官方年号目录
    for split, dirname in (
        ("train", "train2017"),
        ("val", "val2017"),
        ("test", "test2017"),
    ):
        p = src_root / "images" / dirname
        if split not in found and p.is_dir():
            found[split] = p
    # Roboflow 布局回退
    for split, alt in (
        ("train", src_root / "train" / "images"),
        ("val", src_root / "valid" / "images"),
        ("val", src_root / "val" / "images"),
        ("test", src_root / "test" / "images"),
    ):
        if split not in found and alt.is_dir():
            found[split] = alt
    # 仅有 val（如 coco val2017）：先登记为 val，合并时再按 val_fraction 拆出 train（不泄漏）
    if "train" not in found and "val" in found:
        found["_single_pool"] = found["val"]  # 标记：需拆分
        # 仍保留 val 键供发现；实际写入由 merge 按拆分结果决定
    return found


def _list_images(img_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def _progress_iter(
    items: list[Any],
    *,
    desc: str,
    every: int = 2000,
):
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
            print(f"[merge]   {desc}: {i}/{total} ({pct}%)", flush=True)
        yield item


def _split_pool(
    images: list[Path],
    *,
    val_fraction: float,
    seed: int,
) -> dict[str, list[Path]]:
    """将单池图片无重叠拆成 train/val。"""
    if not images:
        return {"train": [], "val": []}
    frac = min(max(float(val_fraction), 0.0), 0.9)
    n = len(images)
    n_val = max(1, int(round(n * frac))) if n >= 2 and frac > 0 else (1 if n == 1 and frac >= 0.5 else 0)
    if n >= 2:
        n_val = min(n_val, n - 1)  # 至少留 1 张 train
    rng = random.Random(seed)
    order = list(images)
    rng.shuffle(order)
    val = order[:n_val]
    train = order[n_val:]
    return {"train": train, "val": val}


def _normalize_split_ratios(ratios: list[float] | tuple[float, ...] | None) -> tuple[float, float, float]:
    """返回 (train, val, test) 比例，和为 1。"""
    if ratios is None:
        return DEFAULT_SPLIT_RATIOS
    vals = [float(x) for x in ratios]
    if len(vals) != 3:
        raise ValueError(f"split_ratios must be [train, val, test], got {ratios!r}")
    if any(v < 0 for v in vals):
        raise ValueError(f"split_ratios must be non-negative, got {ratios!r}")
    s = sum(vals)
    if s <= 0:
        raise ValueError(f"split_ratios sum must be > 0, got {ratios!r}")
    return (vals[0] / s, vals[1] / s, vals[2] / s)


def _split_pool_ratios(
    images: list[Path],
    *,
    ratios: list[float] | tuple[float, ...] | None,
    seed: int,
) -> dict[str, list[Path]]:
    """将图池无重叠划分为 train/val/test（默认 7:2:1）。"""
    r_train, r_val, r_test = _normalize_split_ratios(ratios)
    if not images:
        return {"train": [], "val": [], "test": []}
    rng = random.Random(seed)
    order = list(images)
    rng.shuffle(order)
    n = len(order)
    if n == 1:
        return {"train": order, "val": [], "test": []}
    if n == 2:
        return {"train": [order[0]], "val": [order[1]], "test": []}

    n_test = int(round(n * r_test))
    n_val = int(round(n * r_val))
    # 保证三份都能非空（n>=3），且不超出
    n_test = max(1, n_test) if r_test > 0 else 0
    n_val = max(1, n_val) if r_val > 0 else 0
    if n_test + n_val >= n:
        # 压缩：优先保证至少 1 张 train
        overflow = n_test + n_val - (n - 1)
        if n_test >= n_val and n_test > 0:
            n_test = max(0, n_test - overflow)
        else:
            n_val = max(0, n_val - overflow)
        if n_test + n_val >= n:
            n_test = max(0, min(n_test, 1))
            n_val = max(0, min(n_val, n - 1 - n_test))
    n_train = n - n_val - n_test
    test = order[:n_test]
    val = order[n_test : n_test + n_val]
    train = order[n_test + n_val :]
    assert len(train) == n_train
    return {"train": train, "val": val, "test": test}


def _collect_pool_images(roots: list[Path]) -> list[Path]:
    """从多个数据集根收集全部可发现 split 的图片（去重按 resolve 路径）。"""
    seen: set[Path] = set()
    out: list[Path] = []
    for root in roots:
        print(f"[merge]   scan root={root} ...", flush=True)
        splits = _discover_splits(root)
        if not splits:
            print(f"[merge]   (no splits under {root})", flush=True)
            continue
        for key, img_dir in splits.items():
            if key.startswith("_"):
                continue
            print(f"[merge]   listing {key}: {img_dir} ...", flush=True)
            imgs = _list_images(img_dir)
            print(f"[merge]   listed {len(imgs)} images ({key})", flush=True)
            for img in _progress_iter(imgs, desc=f"dedup {key}", every=10000):
                rp = img.resolve()
                if rp in seen:
                    continue
                seen.add(rp)
                out.append(img)
    return out


def _labels_dir_for_images(img_dir: Path) -> Path:
    if img_dir.name == "images":
        return img_dir.parent / "labels"
    if img_dir.parent.name == "images":
        return img_dir.parent.parent / "labels" / img_dir.name
    return img_dir.parent / "labels"


def load_out_names(data_yaml: Path) -> list[str]:
    """从已有合并集 data.yaml 读取有序 names。"""
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = cfg.get("names")
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(int(x) for x in names)]
    if isinstance(names, list):
        return [str(n) for n in names]
    raise ValueError(f"bad names in {data_yaml}")


def assert_names_prefix_extension(old_names: list[str], new_names: list[str]) -> None:
    """new_names 必须以 old_names 为严格前缀（只允许末尾追加类）。"""
    if len(new_names) < len(old_names):
        raise ValueError(
            f"append names shorter than existing: new={new_names!r} old={old_names!r}"
        )
    for i, (a, b) in enumerate(zip(old_names, new_names)):
        if a != b:
            raise ValueError(
                f"append names must keep existing ids; mismatch at index {i}: "
                f"existing={a!r} new={b!r} (old={old_names!r} new={new_names!r})"
            )


def _path_exists(p: Path) -> bool:
    return p.exists() or p.is_symlink()


def _split_has_images(split_dir: Path) -> bool:
    if not split_dir.is_dir():
        return False
    return any(
        p.is_file() or p.is_symlink()
        for p in split_dir.iterdir()
        if p.suffix.lower() in IMG_EXTS or p.is_symlink()
    )


def _write_out_data_yaml(out_root: Path, names: list[str]) -> dict[str, Any]:
    out_cfg: dict[str, Any] = {
        "path": str(out_root.resolve()),
        "names": {i: n for i, n in enumerate(names)},
    }
    for split in ("train", "val", "test"):
        if _split_has_images(out_root / "images" / split):
            out_cfg[split] = f"images/{split}"
    if "train" not in out_cfg and "val" in out_cfg:
        out_cfg["train"] = "images/val"
    if "val" not in out_cfg and "train" in out_cfg:
        out_cfg["val"] = "images/train"
    (out_root / "data.yaml").write_text(
        yaml.safe_dump(out_cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return out_cfg


def _prefixes_from_specs(specs: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for spec in specs:
        prefix = str(spec.get("prefix") or "").strip()
        if not prefix and spec.get("root"):
            prefix = Path(spec["root"]).name
        if not prefix:
            raise ValueError("remove requires a prefix on each set (or --prefix)")
        if prefix not in seen:
            seen.add(prefix)
            out.append(prefix)
    if not out:
        raise ValueError("remove requires at least one prefix")
    return out


def _file_matches_prefix(name: str, prefixes: list[str]) -> bool:
    return any(name.startswith(p + "_") for p in prefixes)


def remove_by_prefixes(out_root: Path, prefixes: list[str]) -> dict[str, Any]:
    """按输出文件名前缀删除已合并文件；不改 names，不碰其它 prefix。"""
    out_root = Path(out_root)
    data_yaml_out = out_root / "data.yaml"
    if not data_yaml_out.is_file():
        raise FileNotFoundError(
            f"remove requires existing {data_yaml_out}; run a full merge first"
        )
    prefixes = [str(p).strip() for p in prefixes if str(p).strip()]
    if not prefixes:
        raise ValueError("remove requires at least one prefix")
    names = load_out_names(data_yaml_out)
    print(f"[merge] mode=remove out={out_root} prefixes={prefixes}", flush=True)

    images_removed = {s: 0 for s in ("train", "val", "test")}
    labels_removed = {s: 0 for s in ("train", "val", "test")}
    for split in ("train", "val", "test"):
        for folder, counter in (("images", images_removed), ("labels", labels_removed)):
            d = out_root / folder / split
            if not d.is_dir():
                continue
            print(f"[merge]   scanning {folder}/{split} ...", flush=True)
            entries = list(d.iterdir())
            print(
                f"[merge]   {folder}/{split}: {len(entries)} entries, filtering prefixes={prefixes}",
                flush=True,
            )
            for p in _progress_iter(
                entries,
                desc=f"remove {folder}/{split}",
                every=max(2000, min(10000, len(entries) // 20 or 2000)),
            ):
                if not _file_matches_prefix(p.name, prefixes):
                    continue
                p.unlink()
                counter[split] += 1
        print(
            f"[merge]   removed split={split} images={images_removed[split]} "
            f"labels={labels_removed[split]}",
            flush=True,
        )

    _write_out_data_yaml(out_root, names)

    meta_path = out_root / "merge_meta.json"
    prev_meta: dict[str, Any] = {}
    if meta_path.is_file():
        prev_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    drop = set(prefixes)

    def _keep_set(item: dict[str, Any]) -> bool:
        return str(item.get("prefix") or "") not in drop

    hist_sets = [s for s in (prev_meta.get("sets") or []) if _keep_set(s)]
    appends_in = prev_meta.get("appends") if isinstance(prev_meta.get("appends"), list) else []
    appends_out: list[dict[str, Any]] = []
    for a in appends_in:
        if not isinstance(a, dict):
            continue
        kept = [s for s in (a.get("sets") or []) if _keep_set(s)]
        if not kept:
            continue
        item = dict(a)
        item["sets"] = kept
        appends_out.append(item)

    last_remove = {
        "prefixes": prefixes,
        "images_removed": images_removed,
        "labels_removed": labels_removed,
    }
    meta = {
        "out": str(out_root),
        "names": names,
        "mode": "remove",
        "images_removed": images_removed,
        "labels_removed": labels_removed,
        "sets": hist_sets,
        "last_remove": last_remove,
    }
    if appends_out:
        meta["appends"] = appends_out
        meta["last_append"] = appends_out[-1]
    elif prev_meta.get("last_append") and all(
        _keep_set(s) for s in (prev_meta.get("last_append") or {}).get("sets") or []
    ):
        meta["last_append"] = prev_meta["last_append"]
    (out_root / "merge_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return meta


def merge_sets(
    specs: list[dict[str, Any]],
    out_root: Path,
    names: list[str],
    *,
    copy_images: bool = False,
    keep_empty: bool = False,
    mode: str = "full",
) -> dict[str, Any]:
    """
    specs 项：
      - root: Path/str  子集根（含 data.yaml）
      - extra_roots: list  额外根目录（如图片来自 coco_train2017 + coco_val2017）
      - prefix: str     输出文件名前缀（防碰撞）
      - keep: list[str] 保留的源类名（默认 = 与全局 names 交集）
      - split_ratios: [train, val, test]  把本集（+extra_roots）全部图合成一池后
        按比例无重叠划分；默认 [0.7, 0.2, 0.1]。有此项时优先于 val_fraction/assign_split
      - val_fraction: float  仅当源集只有一个图池且未设 split_ratios 时，
        从中划出验证比例（默认 0.2），其余进 train；train/val 不重叠
      - split_seed: int  划分随机种子（默认 0）
      - assign_split: train|val|test  单池时整池写入该 split（忽略 val_fraction）；
        与 split_ratios 互斥（split_ratios 优先）

    mode:
      - full（默认）：清空 out 后全量重建
      - append：保留已有 train/val/test；只写入本 specs；names 须为现有前缀扩展
      - remove：按 specs 的 prefix 删除已写入文件；不改 names
    """
    mode = str(mode or "full").strip().lower()
    if mode not in ("full", "append", "remove"):
        raise ValueError(f"mode must be 'full', 'append' or 'remove', got {mode!r}")
    out_root = Path(out_root)
    if mode == "remove":
        return remove_by_prefixes(out_root, _prefixes_from_specs(specs))
    if not names:
        raise ValueError("global names empty")
    prev_meta: dict[str, Any] | None = None
    if mode == "append":
        data_yaml_out = out_root / "data.yaml"
        if not data_yaml_out.is_file():
            raise FileNotFoundError(
                f"append requires existing {data_yaml_out}; run a full merge first"
            )
        old_names = load_out_names(data_yaml_out)
        assert_names_prefix_extension(old_names, names)
        meta_path = out_root / "merge_meta.json"
        if meta_path.is_file():
            prev_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        out_root.mkdir(parents=True, exist_ok=True)
        print(f"[merge] mode=append out={out_root}", flush=True)
    else:
        if out_root.exists():
            print(f"[merge] removing existing {out_root} ...", flush=True)
            shutil.rmtree(out_root)
        out_root.mkdir(parents=True, exist_ok=True)
        print(f"[merge] out={out_root}", flush=True)

    images_written = {s: 0 for s in ("train", "val", "test")}
    kept_boxes = dropped_boxes = 0
    images_skipped_empty = 0
    set_stats: list[dict[str, Any]] = []

    for si, spec in enumerate(specs):
        src_root = Path(spec["root"])
        extra = [Path(p) for p in (spec.get("extra_roots") or [])]
        all_roots = [src_root, *extra]
        prefix = str(spec.get("prefix") or src_root.name)
        print(
            f"[merge] set {si + 1}/{len(specs)} prefix={prefix} root={src_root}"
            + (f" +{len(extra)} extra" if extra else ""),
            flush=True,
        )
        data_yaml = src_root / "data.yaml"
        if not data_yaml.is_file():
            raise FileNotFoundError(f"missing {data_yaml}")
        src_names = load_source_names(data_yaml)
        keep = spec.get("keep")
        if keep is None:
            keep = [n for n in names if n in set(src_names.values())]
        id_map = build_id_map(src_names, names, list(keep))

        split_plan: dict[str, list[Path]] = {}
        pool_lbl: Path | None = None
        seed = int(spec.get("split_seed", 0))
        use_ratios = "split_ratios" in spec and spec.get("split_ratios") is not None

        if use_ratios:
            print(f"[merge]   pooling images from {len(all_roots)} root(s) ...", flush=True)
            pool = _collect_pool_images(all_roots)
            if not pool:
                raise FileNotFoundError(f"no images under roots {all_roots}")
            print(
                f"[merge]   pooled={len(pool)} → split_ratios={spec.get('split_ratios')}",
                flush=True,
            )
            split_plan = _split_pool_ratios(
                pool,
                ratios=spec.get("split_ratios"),
                seed=seed,
            )
            pool_lbl = None  # 每张图各自推 labels 目录
        else:
            splits = _discover_splits(src_root)
            if not splits:
                raise FileNotFoundError(f"no images splits under {src_root}")
            # 单池（仅 val2017 等）：拆成无重叠 train/val，或 assign_split 整池归位
            if "_single_pool" in splits:
                pool_dir = splits["_single_pool"]
                assign = spec.get("assign_split")
                if assign in ("train", "val", "test"):
                    split_plan = {assign: _list_images(pool_dir)}
                else:
                    val_fraction = float(spec.get("val_fraction", DEFAULT_VAL_FRACTION))
                    split_plan = _split_pool(
                        _list_images(pool_dir),
                        val_fraction=val_fraction,
                        seed=seed,
                    )
                pool_lbl = _labels_dir_for_images(pool_dir)
            else:
                for split, img_dir in splits.items():
                    if split.startswith("_"):
                        continue
                    split_plan[split] = _list_images(img_dir)

        local_kept = local_dropped = local_imgs = 0
        split_counts: dict[str, int] = {}
        for split, imgs in split_plan.items():
            if split.startswith("_") or not imgs:
                continue
            out_img = out_root / "images" / split
            out_lbl = out_root / "labels" / split
            out_img.mkdir(parents=True, exist_ok=True)
            out_lbl.mkdir(parents=True, exist_ok=True)

            n_split = 0
            n_total = len(imgs)
            n_seen = 0
            print(f"[merge]   writing split={split} n={n_total} ...", flush=True)
            for img in _progress_iter(
                imgs, desc=f"write {split}", every=max(500, min(5000, n_total // 20 or 500))
            ):
                n_seen += 1
                # 标签目录：单池共用 pool_lbl；多源/重划分按每张图所在 images 目录推
                if pool_lbl is not None:
                    lbl_dir = pool_lbl
                else:
                    lbl_dir = _labels_dir_for_images(img.parent)
                stem = f"{prefix}_{img.stem}"
                lbl_src = lbl_dir / f"{img.stem}.txt"
                lbl_dst = out_lbl / f"{stem}.txt"
                img_dst = out_img / f"{stem}{img.suffix.lower()}"
                if mode == "append" and (_path_exists(img_dst) or _path_exists(lbl_dst)):
                    raise FileExistsError(
                        f"append collision: {img_dst.name} already under {out_img} "
                        f"(or label exists). Use a different prefix or remove the file."
                    )
                k, d = remap_label_file(lbl_src, lbl_dst, id_map)
                kept_boxes += k
                dropped_boxes += d
                local_kept += k
                local_dropped += d
                if k == 0 and not keep_empty:
                    lbl_dst.unlink(missing_ok=True)
                    images_skipped_empty += 1
                    continue
                _link_or_copy(img, img_dst, copy_images)
                images_written[split] += 1
                local_imgs += 1
                n_split += 1
            print(
                f"[merge]     {split}: kept={n_split}/{n_total} "
                f"(scanned={n_seen}, skipped_empty={n_total - n_split})",
                flush=True,
            )
            split_counts[split] = n_split
        print(
            f"[merge]   done prefix={prefix} images={local_imgs} "
            f"kept_boxes={local_kept} dropped_boxes={local_dropped}",
            flush=True,
        )

        set_stats.append(
            {
                "root": str(src_root),
                "extra_roots": [str(p) for p in extra],
                "prefix": prefix,
                "keep": list(keep),
                "id_map": {str(k): v for k, v in id_map.items()},
                "images": local_imgs,
                "split_counts": split_counts,
                "kept_boxes": local_kept,
                "dropped_boxes": local_dropped,
                "val_fraction": spec.get("val_fraction"),
                "split_ratios": list(spec["split_ratios"]) if use_ratios else None,
                "split_seed": spec.get("split_seed"),
                "assign_split": spec.get("assign_split"),
            }
        )

    out_cfg = _write_out_data_yaml(out_root, names)
    last_append = {
        "kept_boxes": kept_boxes,
        "dropped_boxes": dropped_boxes,
        "images_written": images_written,
        "images_skipped_empty": images_skipped_empty,
        "sets": set_stats,
    }
    if mode == "append" and prev_meta is not None:
        hist_sets = list(prev_meta.get("sets") or [])
        hist_sets.extend(set_stats)
        meta = {
            "out": str(out_root),
            "names": names,
            "mode": "append",
            "kept_boxes": kept_boxes,
            "dropped_boxes": dropped_boxes,
            "images_written": images_written,
            "images_skipped_empty": images_skipped_empty,
            "sets": hist_sets,
            "last_append": last_append,
        }
        if "appends" in prev_meta and isinstance(prev_meta["appends"], list):
            meta["appends"] = list(prev_meta["appends"]) + [last_append]
        else:
            meta["appends"] = [last_append]
    else:
        meta = {
            "out": str(out_root),
            "names": names,
            "mode": mode,
            "kept_boxes": kept_boxes,
            "dropped_boxes": dropped_boxes,
            "images_written": images_written,
            "images_skipped_empty": images_skipped_empty,
            "sets": set_stats,
        }
    (out_root / "merge_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return meta


def load_merge_config(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = list(raw["names"])
    specs = []
    for s in raw["sets"]:
        item: dict[str, Any] = {
            "root": Path(str(s["root"])).expanduser(),
            "prefix": s.get("prefix"),
            "keep": s.get("keep"),
        }
        if "extra_roots" in s and s["extra_roots"] is not None:
            item["extra_roots"] = [Path(str(p)).expanduser() for p in s["extra_roots"]]
        if "val_fraction" in s:
            item["val_fraction"] = float(s["val_fraction"])
        if "split_ratios" in s and s["split_ratios"] is not None:
            item["split_ratios"] = [float(x) for x in s["split_ratios"]]
        if "split_seed" in s:
            item["split_seed"] = int(s["split_seed"])
        if "assign_split" in s and s["assign_split"] is not None:
            item["assign_split"] = str(s["assign_split"])
        specs.append(item)
    return names, specs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, help="merge yaml（names + sets）")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--names",
        default="",
        help="逗号分隔全局类名（无 --config 时必填）",
    )
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="ROOT[:PREFIX[:keep,keep]]",
        help="可重复；例 --set /path/fog_fire_yolo:fire:flame,smoke",
    )
    ap.add_argument("--copy-images", action="store_true")
    ap.add_argument("--keep-empty", action="store_true")
    ap.add_argument(
        "--mode",
        choices=("full", "append", "remove"),
        default="full",
        help="full=清空 out 重建；append=只并入本 config 的 sets；remove=按 prefix 删除已写入文件",
    )
    ap.add_argument(
        "--prefix",
        action="append",
        default=[],
        help="remove 模式下要删除的输出前缀，可重复；也可只用 --config 里的 sets.prefix",
    )
    args = ap.parse_args(argv)

    if args.mode == "remove":
        if args.config:
            _names, specs = load_merge_config(args.config)
        elif args.prefix:
            specs = [{"prefix": p} for p in args.prefix]
        elif args.set:
            specs = []
            for item in args.set:
                parts = item.split(":")
                root = Path(parts[0]).expanduser()
                prefix = parts[1] if len(parts) > 1 and parts[1] else root.name
                specs.append({"prefix": prefix})
        else:
            raise SystemExit("remove 需要 --config、--prefix 或 --set")
        if args.prefix:
            extra = [{"prefix": p} for p in args.prefix]
            specs = extra + specs
        meta = merge_sets(specs, args.out, [], mode="remove")
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        print(f"[ok] removed prefixes from {args.out}")
        return 0

    if args.config:
        names, specs = load_merge_config(args.config)
    else:
        if not args.names.strip() or not args.set:
            raise SystemExit("无 --config 时需要 --names 与至少一个 --set")
        names = [n.strip() for n in args.names.split(",") if n.strip()]
        specs = []
        for item in args.set:
            parts = item.split(":")
            root = Path(parts[0]).expanduser()
            prefix = parts[1] if len(parts) > 1 and parts[1] else root.name
            keep = (
                [x.strip() for x in parts[2].split(",") if x.strip()]
                if len(parts) > 2 and parts[2]
                else None
            )
            specs.append({"root": root, "prefix": prefix, "keep": keep})

    # 相对工程根解析
    for s in specs:
        r = Path(s["root"])
        if not r.is_absolute():
            s["root"] = (ROOT / r).resolve()
        else:
            s["root"] = r.resolve()

    meta = merge_sets(
        specs,
        args.out,
        names,
        copy_images=args.copy_images,
        keep_empty=args.keep_empty,
        mode=args.mode,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[ok] wrote {args.out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
