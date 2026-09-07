#!/usr/bin/env python3
"""从 YOLO 联合集按类别 / prefix 抽图，每组输出两张拼图：无标签 raw + 带框 labeled。

示例：
  # 交互：选类后，再按 merge_meta 提示选 prefix（如 cigarette_or_smoke → cig/cig2）
  python datasets/scripts/preview_yolo_dataset.py \\
    --root /media/.../datasets/detect_w1_fire \\
    --out /tmp/detect_w1_preview \\
    --classes cigarette_or_smoke --mode class

  # 非交互：指定类 + prefix
  python datasets/scripts/preview_yolo_dataset.py \\
    --root .../detect_w1_fire --out /tmp/p \\
    --classes cigarette_or_smoke --prefixes cig2 --mode class -y

  # 全部类/全部相关 prefix（无提示）
  python datasets/scripts/preview_yolo_dataset.py \\
    --root .../detect_w1_fire --out /tmp/p -y
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml
from PIL import Image, ImageDraw, ImageFont

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# 固定调色板（RGB），按 class id 取模
# 高饱和度色，缩略拼图后仍易辨认
_PALETTE = [
    (0, 255, 0),
    (255, 40, 40),
    (40, 140, 255),
    (255, 220, 0),
    (220, 60, 255),
    (0, 255, 255),
    (255, 140, 0),
    (180, 255, 40),
    (255, 40, 180),
    (80, 80, 255),
    (40, 255, 160),
    (255, 255, 40),
    (200, 100, 40),
    (40, 220, 140),
    (240, 240, 240),
]


def load_merge_meta(root: Path) -> dict[str, Any] | None:
    path = Path(root) / "merge_meta.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def prefixes_for_classes(
    merge_meta: dict[str, Any], classes: Iterable[str]
) -> dict[str, list[str]]:
    """根据 merge_meta.sets[].keep 反查每个类对应的 prefix 列表（保序去重）。"""
    wanted = list(classes)
    out: dict[str, list[str]] = {c: [] for c in wanted}
    seen: dict[str, set[str]] = {c: set() for c in wanted}
    for s in merge_meta.get("sets") or []:
        pref = s.get("prefix")
        if not pref:
            continue
        keep = set(s.get("keep") or [])
        for c in wanted:
            if c in keep and pref not in seen[c]:
                seen[c].add(pref)
                out[c].append(str(pref))
    return out


def parse_menu_choice(text: str, opts: list[str]) -> list[str]:
    """解析菜单输入：'1,3' / 'a' / 'all' / '2' → 选中的 opts 子集。"""
    raw = (text or "").strip().lower()
    if not raw:
        raise ValueError("empty choice")
    if raw in ("a", "all", "*"):
        return list(opts)
    picked: list[str] = []
    seen: set[str] = set()
    for part in raw.replace(" ", "").split(","):
        if not part:
            continue
        if not part.isdigit():
            raise ValueError(f"invalid choice token: {part!r}")
        idx = int(part)
        if idx < 1 or idx > len(opts):
            raise ValueError(f"choice out of range: {idx} (1..{len(opts)})")
        item = opts[idx - 1]
        if item not in seen:
            seen.add(item)
            picked.append(item)
    if not picked:
        raise ValueError("no options selected")
    return picked


def prompt_menu(
    opts: list[str],
    *,
    title: str,
    input_fn=input,
) -> list[str]:
    """打印编号菜单，循环直到合法选择。"""
    if not opts:
        return []
    if len(opts) == 1:
        print(f"{title}\n  仅一项，自动选择: {opts[0]}")
        return list(opts)
    while True:
        print(title)
        for i, o in enumerate(opts, 1):
            print(f"  [{i}] {o}")
        print("  [a] all（全部）")
        try:
            choice = input_fn("选择编号（逗号分隔）或 a/all: ").strip()
            return parse_menu_choice(choice, opts)
        except ValueError as e:
            print(f"  无效输入: {e}，请重试")


def resolve_prefixes_interactive(
    root: Path,
    classes: list[str] | None,
    prefixes: list[str] | None,
    *,
    yes: bool = False,
    input_fn=input,
) -> list[str] | None:
    """未指定 --prefixes 时，按 merge_meta / 类提示选择；--yes 则取全部相关。

    返回 None 表示不按 prefix 过滤（预览全部）。
    """
    if prefixes is not None:
        return prefixes

    meta = load_merge_meta(root)
    if classes:
        if meta:
            per = prefixes_for_classes(meta, classes)
            # 并集保序
            opts: list[str] = []
            seen: set[str] = set()
            for c in classes:
                for p in per.get(c) or []:
                    if p not in seen:
                        seen.add(p)
                        opts.append(p)
            hint_lines = [f"  {c}: {', '.join(per[c]) or '(无)'}" for c in classes]
            title = "按类可选 prefix（来自 merge_meta.json）：\n" + "\n".join(hint_lines)
        else:
            # 无 meta：从文件名扫出的 prefix 无法在此获知，返回 None=不过滤
            print("[warn] 无 merge_meta.json，跳过 prefix 选择（按类抽全部）")
            return None
        if not opts:
            print("[warn] 所选类在 merge_meta 中无对应 prefix，按类抽全部")
            return None
        if yes:
            print(f"[yes] 使用全部相关 prefix: {', '.join(opts)}")
            return opts
        return prompt_menu(opts, title=title, input_fn=input_fn)

    # 未指定类：可选全部 sets 的 prefix
    if meta:
        opts = []
        seen = set()
        for s in meta.get("sets") or []:
            p = s.get("prefix")
            if p and p not in seen:
                seen.add(p)
                opts.append(str(p))
    else:
        return None
    if yes:
        print(f"[yes] 使用全部 prefix: {', '.join(opts)}")
        return opts
    return prompt_menu(
        opts,
        title="选择要预览的 prefix（来自 merge_meta.json）：",
        input_fn=input_fn,
    )


def load_names(data_yaml: Path) -> dict[int, str]:
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = cfg.get("names")
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    raise ValueError(f"bad names in {data_yaml}")


def prefixes_from_merge_meta(merge_meta: dict[str, Any] | None) -> list[str]:
    """收集 merge_meta 里登记过的全部 prefix（sets / appends）。"""
    out: list[str] = []
    seen: set[str] = set()

    def add(pref: Any) -> None:
        if not pref:
            return
        p = str(pref)
        if p not in seen:
            seen.add(p)
            out.append(p)

    if not merge_meta:
        return out
    for s in merge_meta.get("sets") or []:
        add(s.get("prefix") if isinstance(s, dict) else None)
    for s in merge_meta.get("appends") or []:
        add(s.get("prefix") if isinstance(s, dict) else None)
    last = merge_meta.get("last_append")
    if isinstance(last, dict):
        add(last.get("prefix"))
    return out


def stem_matches_prefix(stem: str, prefix: str) -> bool:
    """文件名是否属于该 merge prefix（prefix 可含下划线，如 coco_pet）。"""
    return bool(prefix) and (stem == prefix or stem.startswith(prefix + "_"))


def prefix_from_stem(stem: str, known: Iterable[str] | None = None) -> str:
    """从文件名解析 merge prefix。

    有 known 时取最长匹配（coco_pet_0001 → coco_pet，而不是 coco）；
    否则退回第一个 '_' 之前（兼容无 merge_meta 的旧行为）。
    """
    known_list = [p for p in (known or []) if p]
    if known_list:
        hits = [p for p in known_list if stem_matches_prefix(stem, p)]
        if hits:
            return max(hits, key=len)
    if "_" not in stem:
        return stem
    return stem.split("_", 1)[0]


def yolo_to_xyxy(
    xc: float, yc: float, nw: float, nh: float, w: int, h: int
) -> tuple[float, float, float, float]:
    bw, bh = nw * w, nh * h
    x0 = xc * w - bw / 2.0
    y0 = yc * h - bh / 2.0
    return x0, y0, x0 + bw, y0 + bh


def parse_label_file(path: Path) -> list[tuple[int, float, float, float, float]]:
    if not path.is_file():
        return []
    out: list[tuple[int, float, float, float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        xc, yc, nw, nh = map(float, parts[1:5])
        out.append((cid, xc, yc, nw, nh))
    return out


def box_area_ok(nw: float, nh: float, *, min_a: float, max_a: float) -> bool:
    a = nw * nh
    return min_a <= a <= max_a


def discover_split_images(root: Path, split: str) -> list[Path]:
    img_dir = root / "images" / split
    if not img_dir.is_dir():
        raise FileNotFoundError(f"missing {img_dir}")
    return sorted(
        p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def label_path_for_image(img: Path) -> Path:
    # .../images/val/x.jpg → .../labels/val/x.txt
    return img.parent.parent.parent / "labels" / img.parent.name / f"{img.stem}.txt"


def sample_paths(
    candidates: list[Path],
    *,
    n: int,
    seed: int,
) -> list[Path]:
    if not candidates:
        return []
    rng = random.Random(seed)
    order = list(candidates)
    rng.shuffle(order)
    return order[: min(n, len(order))]


def draw_boxes(
    im: Image.Image,
    boxes: list[tuple[int, float, float, float, float]],
    id_to_name: dict[int, str],
    *,
    highlight: set[int] | None = None,
) -> Image.Image:
    out = im.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    # 按短边加粗：拼图缩略后目标约 3–5px；外圈黑边提高对比
    base_w = max(6, int(round(min(w, h) / 48)))
    label_h = max(16, int(round(min(w, h) / 40)))
    for cid, xc, yc, nw, nh in boxes:
        x0, y0, x1, y1 = yolo_to_xyxy(xc, yc, nw, nh, w, h)
        color = _PALETTE[cid % len(_PALETTE)]
        hi = highlight is None or cid in highlight
        width = base_w if hi else max(2, base_w // 3)
        if not hi:
            color = (140, 140, 140)
        edge = width + max(2, width // 2)
        draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 0), width=edge)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
        name = id_to_name.get(cid, str(cid))
        tw = max(8 * len(name) + 8, label_h * 2)
        if y0 >= label_h:
            ty0, ty1 = y0 - label_h, y0
        else:
            ty0, ty1 = y0, min(h, y0 + label_h)
        draw.rectangle([x0, ty0, x0 + tw, ty1], fill=(0, 0, 0))
        draw.rectangle(
            [x0 + 1, ty0 + 1, x0 + tw - 1, ty1 - 1], fill=color
        )
        draw.text((x0 + 4, ty0 + max(1, (label_h - 12) // 2)), name, fill=(0, 0, 0))
    return out


def make_contact_sheet(
    images: list[Image.Image],
    *,
    cell: int = 256,
    cols: int | None = None,
    title: str = "",
) -> Image.Image:
    if not images:
        blank = Image.new("RGB", (cell, cell), (30, 30, 30))
        return blank
    n = len(images)
    cols = cols or max(1, int(math.ceil(math.sqrt(n))))
    rows = int(math.ceil(n / cols))
    title_h = 28 if title else 0
    sheet = Image.new("RGB", (cols * cell, rows * cell + title_h), (20, 20, 20))
    if title:
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 6), title[:120], fill=(0, 255, 120))
    for i, im in enumerate(images):
        thumb = im.convert("RGB").copy()
        thumb.thumbnail((cell - 8, cell - 8))
        x = (i % cols) * cell + (cell - thumb.size[0]) // 2
        y = title_h + (i // cols) * cell + (cell - thumb.size[1]) // 2
        sheet.paste(thumb, (x, y))
    return sheet


def _collect_index(
    root: Path,
    split: str,
    id_to_name: dict[int, str],
    *,
    min_area: float,
    max_area: float,
    known_prefixes: Iterable[str] | None = None,
) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    """返回 by_class_name -> images, by_prefix -> images。"""
    by_class: dict[str, list[Path]] = defaultdict(list)
    by_prefix: dict[str, list[Path]] = defaultdict(list)
    seen_class: dict[str, set[Path]] = defaultdict(set)
    known = list(known_prefixes) if known_prefixes else None
    for img in discover_split_images(root, split):
        prefix = prefix_from_stem(img.stem, known)
        by_prefix[prefix].append(img)
        boxes = parse_label_file(label_path_for_image(img))
        if not boxes:
            continue
        present: set[int] = set()
        for cid, _xc, _yc, nw, nh in boxes:
            if not box_area_ok(nw, nh, min_a=min_area, max_a=max_area):
                # 仍允许图进入「含该类」池时用宽松：只要有一个合格框就算
                continue
            present.add(cid)
        # 若过滤后为空，退回「任意含该类」（避免稀有类全空）
        if not present:
            present = {cid for cid, *_ in boxes}
        for cid in present:
            name = id_to_name.get(cid, str(cid))
            if img not in seen_class[name]:
                by_class[name].append(img)
                seen_class[name].add(img)
    return dict(by_class), dict(by_prefix)


def _write_pair(
    out_dir: Path,
    group_key: str,
    imgs: list[Path],
    id_to_name: dict[int, str],
    *,
    highlight: set[int] | None,
    cell: int,
    title_prefix: str,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_ims: list[Image.Image] = []
    lab_ims: list[Image.Image] = []
    stems: list[str] = []
    for p in imgs:
        with Image.open(p) as im:
            base = im.convert("RGB")
            boxes = parse_label_file(label_path_for_image(p))
            raw_ims.append(base.copy())
            lab_ims.append(draw_boxes(base, boxes, id_to_name, highlight=highlight))
            stems.append(p.name)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in group_key)
    title = f"{title_prefix}{group_key} n={len(imgs)}"
    raw_path = out_dir / f"{safe}_raw.jpg"
    lab_path = out_dir / f"{safe}_labeled.jpg"
    make_contact_sheet(raw_ims, cell=cell, title=title + " (raw)").save(
        raw_path, quality=85
    )
    make_contact_sheet(lab_ims, cell=cell, title=title + " (labeled)").save(
        lab_path, quality=85
    )
    return {
        "group": group_key,
        "n": len(imgs),
        "files": stems,
        "raw": str(raw_path),
        "labeled": str(lab_path),
    }


def preview_dataset(
    root: Path,
    out_root: Path,
    *,
    split: str = "val",
    per_group: int = 9,
    seed: int = 0,
    modes: Iterable[str] = ("class", "prefix"),
    cell_size: int = 256,
    classes: list[str] | None = None,
    prefixes: list[str] | None = None,
    min_area: float = 0.0,
    max_area: float = 1.0,
) -> dict[str, Any]:
    root = Path(root)
    out_root = Path(out_root)
    data_yaml = root / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"missing {data_yaml}")
    id_to_name = load_names(data_yaml)
    name_to_id = {v: k for k, v in id_to_name.items()}
    merge_meta = load_merge_meta(root)
    known_prefixes = prefixes_from_merge_meta(merge_meta)
    if prefixes:
        for p in prefixes:
            if p not in known_prefixes:
                known_prefixes.append(p)
    by_class, by_prefix = _collect_index(
        root,
        split,
        id_to_name,
        min_area=min_area,
        max_area=max_area,
        known_prefixes=known_prefixes,
    )

    modes_set = set(modes)
    groups_meta: list[dict[str, Any]] = []

    pref_filter = set(prefixes) if prefixes else None

    if "class" in modes_set:
        class_names = classes or sorted(
            by_class.keys(), key=lambda n: name_to_id.get(n, 999)
        )
        for i, cname in enumerate(class_names):
            cands = by_class.get(cname) or []
            if pref_filter is not None:
                cands = [
                    p
                    for p in cands
                    if prefix_from_stem(p.stem, known_prefixes) in pref_filter
                ]
            picked = sample_paths(cands, n=per_group, seed=seed + i * 17)
            hl = {name_to_id[cname]} if cname in name_to_id else None
            group_key = cname
            if pref_filter is not None:
                group_key = f"{cname}__{'-'.join(sorted(pref_filter))}"
            info = _write_pair(
                out_root / "by_class",
                group_key,
                picked,
                id_to_name,
                highlight=hl,
                cell=cell_size,
                title_prefix="class=",
            )
            info["mode"] = "class"
            info["class"] = cname
            info["prefixes"] = sorted(pref_filter) if pref_filter else None
            info["available"] = len(cands)
            groups_meta.append(info)

    if "prefix" in modes_set:
        pref_names = prefixes or sorted(by_prefix.keys())
        for i, pref in enumerate(pref_names):
            cands = by_prefix.get(pref) or []
            picked = sample_paths(cands, n=per_group, seed=seed + 1000 + i * 17)
            info = _write_pair(
                out_root / "by_prefix",
                pref,
                picked,
                id_to_name,
                highlight=None,
                cell=cell_size,
                title_prefix="prefix=",
            )
            info["mode"] = "prefix"
            info["available"] = len(cands)
            groups_meta.append(info)

    meta = {
        "root": str(root),
        "out": str(out_root),
        "split": split,
        "per_group": per_group,
        "seed": seed,
        "modes": list(modes),
        "classes": classes,
        "prefixes": prefixes,
        "names": id_to_name,
        "groups": groups_meta,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "preview_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # 简短 index
    lines = [
        f"# preview {root.name} split={split}",
        "",
        "每组两张：`*_raw.jpg`（无框）与 `*_labeled.jpg`（带框）。",
        "",
    ]
    for g in groups_meta:
        lines.append(
            f"- **{g['mode']} `{g['group']}`** available={g['available']} shown={g['n']}"
        )
        lines.append(f"  - raw: `{g['raw']}`")
        lines.append(f"  - labeled: `{g['labeled']}`")
    (out_root / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="YOLO 联合集根（含 data.yaml）")
    ap.add_argument("--out", type=Path, required=True, help="预览输出目录")
    ap.add_argument("--split", default="val", choices=("train", "val", "test"))
    ap.add_argument("--per-group", type=int, default=9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--mode",
        action="append",
        choices=("class", "prefix"),
        help="可重复；默认同时 by_class + by_prefix",
    )
    ap.add_argument(
        "--classes",
        default="",
        help="类别名，逗号分隔（如 cigarette_or_smoke,ebike）；默认交互选择或全部",
    )
    ap.add_argument(
        "--prefixes",
        default="",
        help="文件名前缀，逗号分隔（如 cig,cig2）；省略则按 merge_meta 交互选择",
    )
    ap.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="非交互：未给 --prefixes/--classes 时自动选全部相关项",
    )
    ap.add_argument("--cell", type=int, default=256)
    ap.add_argument("--min-area", type=float, default=0.0, help="按类候选时框面积下限")
    ap.add_argument("--max-area", type=float, default=1.0, help="按类候选时框面积上限")
    args = ap.parse_args(argv)

    modes = tuple(args.mode) if args.mode else ("class", "prefix")
    classes = [x.strip() for x in args.classes.split(",") if x.strip()] or None
    prefixes_arg = [x.strip() for x in args.prefixes.split(",") if x.strip()] or None

    if classes is None and "class" in modes and not args.yes:
        id_to_name = load_names(args.root / "data.yaml")
        name_opts = [id_to_name[k] for k in sorted(id_to_name)]
        classes = prompt_menu(
            name_opts,
            title="选择要预览的类别（data.yaml names）：",
        )

    prefixes = resolve_prefixes_interactive(
        args.root,
        classes,
        prefixes_arg,
        yes=args.yes,
    )

    meta = preview_dataset(
        args.root,
        args.out,
        split=args.split,
        per_group=args.per_group,
        seed=args.seed,
        modes=modes,
        cell_size=args.cell,
        classes=classes,
        prefixes=prefixes,
        min_area=args.min_area,
        max_area=args.max_area,
    )
    print(
        json.dumps(
            {
                "out": meta["out"],
                "n_groups": len(meta["groups"]),
                "classes": meta.get("classes"),
                "prefixes": meta.get("prefixes"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"[ok] wrote {args.out / 'index.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
