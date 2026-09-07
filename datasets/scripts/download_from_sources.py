#!/usr/bin/env python3
"""按 dataset_sources.yaml 下载 enabled public 源；按 url+local_dir 去重。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from patrol_perception.catalog import (  # noqa: E402
    collect_public_download_jobs,
    load_dataset_sources,
)

COCO_ZIP = {
    "coco8": "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip",
    "coco128": "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip",
}

# COCO 2017：Ultralytics YOLO 标签 zip 内含 train2017 + val2017
# val2017 约 5k 图 / ~1GB；train2017 约 118k 图 / ~18GB
COCO_2017 = {
    "labels": "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
}
COCO_2017_VARIANTS = frozenset({"val2017", "train2017"})
# 兼容旧测试 / 引用
COCO_VAL2017 = {
    "images": COCO_2017["val2017"],
    "labels": COCO_2017["labels"],
}

# COCO80 官方顺序（与 Ultralytics YOLO 标签 id 一致）
COCO80_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

# 文档页 / 搜索页：不可直接 wget，需人工 Clone 后填直链
PAGE_HINTS = (
    "platform.ultralytics.com/explore",
    "universe.roboflow.com/search",
    "roboflow.com/search",
    "docs.ultralytics.com/datasets/",
)

HF_DATASETS_PREFIX = "huggingface.co/datasets/"
HF_TOKEN_FILE = ROOT / ".hf_token"
TACO_REPO_URL = "https://github.com/pedropro/TACO.git"


def _marker(local_dir: Path) -> Path:
    return local_dir / ".download_ok.json"


def _is_page_url(url: str, kind: str) -> bool:
    if kind in ("page", "manual"):
        return True
    if kind in ("coco", "huggingface", "ndjson", "taco"):
        return False
    return any(h in url for h in PAGE_HINTS)


def _is_taco_url(url: str) -> bool:
    u = url.lower()
    return (
        "tacodataset.org" in u
        or "github.com/pedropro/taco" in u
    )


def parse_hf_dataset_repo_id(url: str) -> str:
    """https://huggingface.co/datasets/org/name[/...] → org/name"""
    u = url.strip().rstrip("/")
    key = HF_DATASETS_PREFIX
    if key not in u:
        raise ValueError(f"not a HuggingFace dataset url: {url}")
    rest = u.split(key, 1)[1]
    parts = [p for p in rest.split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"bad HuggingFace dataset url: {url}")
    return f"{parts[0]}/{parts[1]}"


def resolve_ndjson_path(path: str | Path | None) -> Path | None:
    """解析 ndjson 路径：绝对/~ 直接用；相对则依次试 cwd、~/Downloads、工程根。"""
    if path is None or not str(path).strip():
        return None
    p = Path(str(path)).expanduser()
    if p.is_file():
        return p.resolve()
    if p.is_absolute():
        return p
    for base in (Path.cwd(), Path.home() / "Downloads", ROOT):
        cand = (base / p).expanduser()
        if cand.is_file():
            return cand.resolve()
    return (Path.cwd() / p).resolve()


def resolve_hf_token() -> str | None:
    """
    优先级：
    1. 环境变量 HF_TOKEN / HUGGING_FACE_HUB_TOKEN
    2. 工程内 .hf_token（已 gitignore，单行写入 token）
    3. huggingface_hub 已登录缓存
    """
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = (os.environ.get(key) or "").strip()
        if val:
            return val
    if HF_TOKEN_FILE.is_file():
        val = HF_TOKEN_FILE.read_text(encoding="utf-8").strip()
        if val and not val.startswith("#"):
            return val
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def _write_marker(local_dir: Path, meta: dict) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    _marker(local_dir).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def remap_coco_local_dir(local_dir: Path, variant: str) -> Path:
    """yaml 中四类共用 coco_val2017；选 train2017 时改落到 coco_train2017，避免覆盖 val。"""
    local_dir = Path(local_dir)
    if variant != "train2017":
        return local_dir
    name = local_dir.name
    if "train2017" in name:
        return local_dir
    if name == "coco_val2017" or name.endswith("_val2017") or "val2017" in name:
        return local_dir.with_name(name.replace("val2017", "train2017"))
    if name in {"coco", "coco128", "coco8"}:
        return local_dir.with_name("coco_train2017")
    return local_dir.parent / "coco_train2017"


def _write_coco_2017_data_yaml(local_dir: Path, split: str) -> Path:
    """为 val2017 / train2017 写 data.yaml。"""
    import yaml

    if split not in COCO_2017_VARIANTS:
        raise ValueError(f"bad coco split: {split}")
    cfg: dict = {
        "path": str(local_dir.resolve()),
        "names": {i: n for i, n in enumerate(COCO80_NAMES)},
    }
    if split == "val2017":
        # 单池回放：train/val 都指向 val2017（merge 可用 val_fraction 再拆）
        cfg["train"] = "images/val2017"
        cfg["val"] = "images/val2017"
    else:
        cfg["train"] = "images/train2017"
    out = local_dir / "data.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out


def _write_coco_val_data_yaml(local_dir: Path) -> Path:
    """兼容旧名。"""
    return _write_coco_2017_data_yaml(local_dir, "val2017")


def _find_split_dir(extract_root: Path, split: str, *, under_labels: bool) -> Path | None:
    """在解压树中定位 labels/<split> 或 <split>/（images）。"""
    if under_labels:
        for cand in (
            extract_root / "coco" / "labels" / split,
            extract_root / "labels" / split,
        ):
            if cand.is_dir():
                return cand
        found = list(extract_root.rglob(split))
        return next((p for p in found if p.is_dir() and p.parent.name == "labels"), None)
    direct = extract_root / split
    if direct.is_dir():
        return direct
    return next((p for p in extract_root.rglob(split) if p.is_dir()), None)


def download_coco_2017_split(local_dir: Path, split: str, force: bool) -> str:
    """下载 COCO 2017 某一 split 的图片 + Ultralytics YOLO 标签，写 data.yaml。"""
    if split not in COCO_2017_VARIANTS:
        raise ValueError(f"unsupported coco 2017 split: {split}")
    if _marker(local_dir).exists() and not force:
        print(f"[skip] coco {split} already present: {local_dir}")
        return "skip"

    from ultralytics.utils.downloads import safe_download

    local_dir.mkdir(parents=True, exist_ok=True)
    images_url = COCO_2017[split]
    labels_url = COCO_2017["labels"]

    # 1) YOLO labels（zip 内为 coco/labels/{train2017,val2017}/...）
    print(f"[coco] downloading YOLO labels → {local_dir}")
    labels_archive = Path(safe_download(labels_url, dir=local_dir, unzip=False))
    labels_extract = local_dir / "_labels_extract"
    if labels_extract.exists():
        shutil.rmtree(labels_extract)
    labels_extract.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(labels_archive, "r") as zf:
        zf.extractall(labels_extract)
    src_lbl = _find_split_dir(labels_extract, split, under_labels=True)
    if src_lbl is None:
        raise RuntimeError(f"labels zip 中未找到 labels/{split}: {labels_extract}")
    dest_lbl = local_dir / "labels" / split
    if dest_lbl.exists():
        shutil.rmtree(dest_lbl)
    dest_lbl.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src_lbl), str(dest_lbl))
    shutil.rmtree(labels_extract, ignore_errors=True)
    labels_archive.unlink(missing_ok=True)

    # 2) images：{split}.zip → images/{split}/
    print(f"[coco] downloading {split} images (~large for train2017) → {local_dir}")
    img_archive = Path(safe_download(images_url, dir=local_dir, unzip=False))
    img_extract = local_dir / "_images_extract"
    if img_extract.exists():
        shutil.rmtree(img_extract)
    img_extract.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(img_archive, "r") as zf:
        zf.extractall(img_extract)
    src_imgs = _find_split_dir(img_extract, split, under_labels=False)
    if src_imgs is None or not src_imgs.is_dir():
        raise RuntimeError(f"images zip 中未找到 {split}/: {img_extract}")
    dest_imgs = local_dir / "images" / split
    if dest_imgs.exists():
        shutil.rmtree(dest_imgs)
    dest_imgs.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src_imgs), str(dest_imgs))
    shutil.rmtree(img_extract, ignore_errors=True)
    img_archive.unlink(missing_ok=True)

    n_img = sum(1 for p in dest_imgs.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    n_lbl = sum(1 for p in dest_lbl.iterdir() if p.suffix == ".txt")
    yaml_path = _write_coco_2017_data_yaml(local_dir, split)
    _write_marker(
        local_dir,
        {
            "kind": "coco",
            "variant": split,
            "images_url": images_url,
            "labels_url": labels_url,
            "local_dir": str(local_dir),
            "n_images": n_img,
            "n_labels": n_lbl,
            "data_yaml": str(yaml_path),
        },
    )
    print(f"[ok] coco {split} ready at {local_dir} (images={n_img}, labels={n_lbl})")
    return "downloaded"


def download_coco_val2017(local_dir: Path, force: bool) -> str:
    """兼容旧入口。"""
    return download_coco_2017_split(local_dir, "val2017", force)


def download_coco(local_dir: Path, variant: str, force: bool) -> str:
    """下载 COCO：coco8/coco128 子集，或 val2017/train2017。返回 skip|downloaded。"""
    if variant in COCO_2017_VARIANTS:
        return download_coco_2017_split(local_dir, variant, force)
    if variant not in COCO_ZIP:
        raise SystemExit(
            f"unsupported --coco-variant {variant!r}; use one of "
            f"{sorted(set(COCO_ZIP) | set(COCO_2017_VARIANTS))} "
            f"or prepare full COCO manually under {local_dir}"
        )
    if _marker(local_dir).exists() and not force:
        print(f"[skip] coco already present: {local_dir}")
        return "skip"

    from ultralytics.utils.downloads import safe_download

    local_dir.mkdir(parents=True, exist_ok=True)
    zip_url = COCO_ZIP[variant]
    print(f"[coco] downloading {variant} → {local_dir}")
    archive = safe_download(zip_url, dir=local_dir, unzip=False)
    archive = Path(archive)
    extract_root = local_dir / "_extract"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(extract_root)
    # zip 内通常为 coco128/ 或 coco8/
    nested = next(
        (p for p in extract_root.iterdir() if p.is_dir()),
        None,
    )
    if nested is None:
        raise RuntimeError(f"empty extract: {extract_root}")
    for item in nested.iterdir():
        dest = local_dir / item.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        shutil.move(str(item), str(dest))
    shutil.rmtree(extract_root, ignore_errors=True)
    archive.unlink(missing_ok=True)
    _write_marker(
        local_dir,
        {
            "kind": "coco",
            "variant": variant,
            "url": zip_url,
            "local_dir": str(local_dir),
        },
    )
    print(f"[ok] coco {variant} ready at {local_dir}")
    return "downloaded"


def download_huggingface(url: str, local_dir: Path, force: bool) -> str:
    """等价于: hf download <repo_id> --repo-type=dataset --local-dir <local_dir>"""
    if _marker(local_dir).exists() and not force:
        print(f"[skip] huggingface already present: {local_dir}")
        return "skip"

    try:
        from huggingface_hub import login, snapshot_download
    except ImportError as e:
        raise SystemExit(
            "缺少 huggingface_hub，请先: pip install huggingface_hub\n"
            f"原始错误: {e}"
        ) from e

    token = resolve_hf_token()
    if not token:
        raise SystemExit(
            "未配置 Hugging Face token（匿名下载会限速/易失败）。\n"
            "任选其一：\n"
            f"  1) 把 token 单行写入 {HF_TOKEN_FILE}（已 gitignore）\n"
            "  2) export HF_TOKEN=hf_xxx\n"
            "  3) huggingface-cli login\n"
            "Token 在 https://huggingface.co/settings/tokens 创建（Read 权限即可）"
        )

    # 写入本机缓存，后续 CLI/库也能用
    login(token=token, add_to_git_credential=False)
    os.environ.setdefault("HF_TOKEN", token)

    repo_id = parse_hf_dataset_repo_id(url)
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[huggingface] authenticated download {repo_id} → {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        token=token,
    )
    _write_marker(
        local_dir,
        {
            "kind": "huggingface",
            "repo_id": repo_id,
            "url": url,
            "local_dir": str(local_dir),
            "authenticated": True,
        },
    )
    print(f"[ok] huggingface {repo_id} ready at {local_dir}")
    return "downloaded"


def convert_ndjson_dataset(
    ndjson_path: Path,
    local_dir: Path,
    force: bool,
    *,
    source_url: str = "",
    concurrency: int = 8,
    allow_partial: bool = False,
) -> str:
    """
    Ultralytics Platform / Hub 导出的 .ndjson → 本地 YOLO 布局。
    使用本仓库 convert_ndjson_resilient（低并发、可续跑），避免官方版任意图失败即中止且不写 data.yaml。
    """
    if _marker(local_dir).exists() and not force:
        print(f"[skip] ndjson already converted: {local_dir}")
        return "skip"
    if not ndjson_path.is_file():
        raise SystemExit(
            f"找不到 ndjson: {ndjson_path}\n"
            "请先从 Ultralytics Platform 下载 .ndjson，或在 yaml 写 ndjson: ~/Downloads/xxx.ndjson\n"
            "或 CLI: --ndjson /path/to/file.ndjson"
        )

    import asyncio
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        "convert_ndjson_resilient",
        Path(__file__).with_name("convert_ndjson_resilient.py"),
    )
    if _spec is None or _spec.loader is None:
        raise SystemExit("无法加载 convert_ndjson_resilient.py")
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ndjson] {ndjson_path} → {local_dir} (concurrency={concurrency})")
    out = asyncio.run(
        _mod.convert_ndjson_resilient(
            ndjson_path,
            local_dir,
            concurrency=concurrency,
            allow_partial=allow_partial,
        )
    )
    if not allow_partial:
        _write_marker(
            local_dir,
            {
                "kind": "ndjson",
                "ndjson": str(ndjson_path),
                "url": source_url,
                "local_dir": str(local_dir),
                "converted_to": str(out),
            },
        )
    print(f"[ok] ndjson converted → {out}")
    return "downloaded"


def _ensure_taco_repo(local_dir: Path, repo_url: str = TACO_REPO_URL) -> Path:
    """确保 local_dir 是 TACO git 仓（含 data/annotations.json）。"""
    ann = local_dir / "data" / "annotations.json"
    if ann.is_file():
        return ann
    local_dir.mkdir(parents=True, exist_ok=True)
    if any(local_dir.iterdir()):
        # 目录非空但缺标注：仍尝试在子目录找
        found = list(local_dir.rglob("annotations.json"))
        if found:
            return found[0]
        raise SystemExit(
            f"TACO local_dir 非空且缺少 data/annotations.json: {local_dir}\n"
            "请清空该目录或换空目录后再跑。"
        )
    import subprocess

    print(f"[taco] git clone {repo_url} → {local_dir}")
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(local_dir)],
        check=True,
    )
    if not ann.is_file():
        raise SystemExit(f"clone 后仍缺少 {ann}")
    return ann


def _taco_fetch_images(
    annotations_path: Path,
    *,
    workers: int = 16,
    prefer_resized: bool = True,
    retries: int = 3,
) -> tuple[int, int, int]:
    """
    从 Flickr/S3 URL 并行拉图到 annotations 同级目录。
    返回 (new, skipped, failed)。
    prefer_resized=True 优先 flickr_640_url（更小更快）；False 用原图 flickr_url。
    """
    try:
        import requests
        from concurrent.futures import ThreadPoolExecutor, as_completed
    except ImportError as e:
        raise SystemExit(
            "TACO 拉图需要 requests：pip install requests\n"
            f"原始错误: {e}"
        ) from e

    dataset_dir = annotations_path.parent
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    images = annotations.get("images") or []
    n = len(images)
    print(
        f"[taco] downloading {n} images into {dataset_dir} "
        f"(workers={workers}, prefer_resized={prefer_resized}, resumable)"
    )

    session_local = threading.local()

    def _session() -> "requests.Session":
        s = getattr(session_local, "session", None)
        if s is None:
            s = requests.Session()
            s.headers.update({"User-Agent": "patrol_perception-taco/1.0"})
            session_local.session = s
        return s

    def _pick_url(image: dict) -> str | None:
        if prefer_resized:
            return (
                image.get("flickr_640_url")
                or image.get("flickr_url")
                or image.get("coco_url")
            )
        return image.get("flickr_url") or image.get("coco_url") or image.get(
            "flickr_640_url"
        )

    def _one(image: dict) -> str:
        file_name = image["file_name"]
        file_path = dataset_dir / file_name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if file_path.is_file() and file_path.stat().st_size > 0:
            return "skip"
        url = _pick_url(image)
        if not url:
            return "fail"
        last_err: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                resp = _session().get(url, timeout=(10, 90))
                resp.raise_for_status()
                if len(resp.content) < 100:
                    raise RuntimeError("empty body")
                tmp = file_path.with_suffix(file_path.suffix + ".part")
                tmp.write_bytes(resp.content)
                tmp.replace(file_path)
                return "new"
            except Exception as exc:
                last_err = exc
                time.sleep(min(2 * attempt, 8))
        print(f"[taco] fail {file_name}: {last_err}")
        return "fail"

    ok = skipped = failed = 0
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(_one, im) for im in images]
        for fut in as_completed(futs):
            status = fut.result()
            done += 1
            if status == "new":
                ok += 1
            elif status == "skip":
                skipped += 1
            else:
                failed += 1
            if done % 50 == 0 or done == n:
                print(
                    f"[taco] progress {done}/{n} "
                    f"(new={ok}, skip={skipped}, fail={failed})"
                )
    return ok, skipped, failed


def download_taco(
    local_dir: Path,
    force: bool,
    *,
    source_url: str = "",
    unofficial: bool = False,
    workers: int = 16,
    prefer_resized: bool = True,
) -> str:
    """
    整合官方流程：clone https://github.com/pedropro/TACO + 按 annotations 拉 Flickr 图。
    litter_* 四类共用同一 local_dir，去重后只下一份。
    """
    if _marker(local_dir).exists() and not force:
        print(f"[skip] taco already present: {local_dir}")
        return "skip"

    ann_name = (
        "annotations_unofficial.json" if unofficial else "annotations.json"
    )
    # clone 官方仓后，指定 annotations 路径
    official_ann = _ensure_taco_repo(local_dir)
    ann_path = official_ann.parent / ann_name
    if not ann_path.is_file():
        raise SystemExit(f"缺少标注文件: {ann_path}")

    new_n, skip_n, fail_n = _taco_fetch_images(
        ann_path, workers=workers, prefer_resized=prefer_resized
    )
    meta = {
        "kind": "taco",
        "url": source_url or TACO_REPO_URL,
        "local_dir": str(local_dir),
        "annotations": str(ann_path),
        "images_new": new_n,
        "images_skipped": skip_n,
        "images_failed": fail_n,
        "prefer_resized": prefer_resized,
    }
    if fail_n == 0:
        _write_marker(local_dir, meta)
        print(f"[ok] taco ready at {local_dir} (new={new_n}, skip={skip_n})")
        return "downloaded"

    # 有失败不写完成标记，便于下次续下
    print(
        f"[partial] taco {local_dir}: new={new_n}, skip={skip_n}, fail={fail_n}。"
        "未写 .download_ok.json，请重新运行同一命令续下失败项。"
    )
    (local_dir / ".download_partial.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return "downloaded"


def download_direct_url(url: str, local_dir: Path, force: bool) -> str:
    if _marker(local_dir).exists() and not force:
        print(f"[skip] already present: {local_dir}")
        return "skip"
    from ultralytics.utils.downloads import safe_download

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[url] {url} → {local_dir}")
    safe_download(url, dir=local_dir, unzip=True)
    _write_marker(local_dir, {"kind": "url", "url": url, "local_dir": str(local_dir)})
    print(f"[ok] {local_dir}")
    return "downloaded"


def run_job(
    job: dict,
    *,
    coco_variant: str,
    force: bool,
    dry_run: bool,
    ndjson_override: Path | None = None,
    taco_workers: int = 16,
    taco_prefer_resized: bool = True,
    ndjson_concurrency: int = 8,
) -> str:
    url = job["url"]
    local_dir: Path = job["local_dir"]
    kind = job.get("kind") or "url"
    if kind == "url" and HF_DATASETS_PREFIX in url:
        kind = "huggingface"
    if kind == "url" and _is_taco_url(url):
        kind = "taco"
    classes = ",".join(job.get("classes") or [])
    print(f"\n=== job classes=[{classes}] kind={kind} ===")
    print(f"    url={url}")

    if kind == "coco" or (
        "docs.ultralytics.com/datasets/detect/coco" in url and kind != "page"
    ):
        remapped = remap_coco_local_dir(local_dir, coco_variant)
        if remapped != local_dir:
            print(f"    local_dir={local_dir} → {remapped} (coco-variant={coco_variant})")
        else:
            print(f"    local_dir={local_dir}")
        if dry_run:
            return "dry_run"
        return download_coco(remapped, coco_variant, force)

    print(f"    local_dir={local_dir}")

    if dry_run:
        if kind == "ndjson" or ndjson_override or job.get("ndjson"):
            ndjp = ndjson_override or resolve_ndjson_path(job.get("ndjson"))
            print(f"    ndjson={ndjp}")
        return "dry_run"

    if kind == "huggingface" or HF_DATASETS_PREFIX in url:
        return download_huggingface(url, local_dir, force)

    if kind == "taco" or _is_taco_url(url):
        unofficial = bool(job.get("unofficial"))
        return download_taco(
            local_dir,
            force,
            source_url=url,
            unofficial=unofficial,
            workers=taco_workers,
            prefer_resized=taco_prefer_resized,
        )

    if kind == "ndjson" or (job.get("ndjson") and kind != "page"):
        ndjp = ndjson_override or resolve_ndjson_path(job.get("ndjson"))
        if ndjp is None:
            raise SystemExit(
                f"kind=ndjson 但未指定文件。yaml 写 ndjson: ~/Downloads/xxx.ndjson "
                f"或传 --ndjson（job={job.get('name')})"
            )
        return convert_ndjson_dataset(
            ndjp,
            local_dir,
            force,
            source_url=url,
            concurrency=ndjson_concurrency,
        )

    if _is_page_url(url, kind):
        print(
            "[manual] 文档/搜索页，无法自动下载。"
            "请 Clone/导出 YOLO 或 Platform .ndjson 后："
            "yaml 设 kind: ndjson + ndjson: 路径，或 --ndjson。"
        )
        return "manual"

    return download_direct_url(url, local_dir, force)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sources",
        type=Path,
        default=ROOT / "config" / "dataset_sources.yaml",
    )
    ap.add_argument(
        "--classes",
        type=str,
        default="",
        help="逗号分隔类名；默认处理 yaml 中全部 enabled public",
    )
    ap.add_argument(
        "--coco-variant",
        choices=sorted(set(COCO_ZIP) | set(COCO_2017_VARIANTS)),
        default="val2017",
        help=(
            "COCO 变体：val2017=小回放（默认）；train2017=全量训练图（~18GB，"
            "落到 coco_train2017）；coco128/coco8=冒烟子集"
        ),
    )
    ap.add_argument(
        "--data-root",
        type=str,
        default="",
        help="数据根（绝对路径可指外置硬盘）；覆盖 yaml roots.data_root / PATROL_DATA_ROOT",
    )
    ap.add_argument(
        "--ndjson",
        type=str,
        default="",
        help="覆盖 yaml 的 ndjson 路径（用于 kind=ndjson 任务）",
    )
    ap.add_argument(
        "--taco-workers",
        type=int,
        default=16,
        help="TACO 并行下载线程数（默认 16）",
    )
    ap.add_argument(
        "--taco-full",
        action="store_true",
        help="TACO 下原图 flickr_url（更慢更大）；默认用 flickr_640_url",
    )
    ap.add_argument(
        "--ndjson-concurrency",
        type=int,
        default=8,
        help="ndjson 拉图并发（默认 8；官方 128 易超时中断）",
    )
    ap.add_argument("--force", action="store_true", help="忽略 .download_ok.json 重下")
    ap.add_argument("--dry-run", action="store_true", help="只打印去重后的任务")
    args = ap.parse_args(argv)

    sources = load_dataset_sources(args.sources)
    class_names = (
        [c.strip() for c in args.classes.split(",") if c.strip()] or None
    )
    data_root = args.data_root.strip() or None
    jobs = collect_public_download_jobs(
        sources, class_names, root=ROOT, data_root=data_root
    )
    if not jobs:
        print("no enabled public download jobs")
        return 0

    ndjson_override = resolve_ndjson_path(args.ndjson) if args.ndjson.strip() else None
    if args.ndjson.strip() and ndjson_override is not None and not ndjson_override.is_file():
        # resolve_ndjson_path 对不存在的绝对路径仍返回 path；再检查
        if not Path(args.ndjson).expanduser().is_file():
            raise SystemExit(f"--ndjson 文件不存在: {args.ndjson}")

    print(f"deduped jobs: {len(jobs)}")
    counts = {"downloaded": 0, "skip": 0, "manual": 0, "dry_run": 0}
    for job in jobs:
        status = run_job(
            job,
            coco_variant=args.coco_variant,
            force=args.force,
            dry_run=args.dry_run,
            ndjson_override=ndjson_override if job.get("kind") == "ndjson" or job.get("ndjson") else None,
            taco_workers=args.taco_workers,
            taco_prefer_resized=not args.taco_full,
            ndjson_concurrency=args.ndjson_concurrency,
        )
        counts[status] = counts.get(status, 0) + 1

    print("\nsummary:", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
