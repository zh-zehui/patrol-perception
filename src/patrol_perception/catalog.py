"""events_catalog：事件使能 → 训练类并集 / 推理门控。"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import yaml

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CATALOG = _ROOT / "config" / "events_catalog.yaml"
_DEFAULT_CLASSES = _ROOT / "config" / "classes.yaml"
_DEFAULT_DATASET_SOURCES = _ROOT / "config" / "dataset_sources.yaml"

ALLOWED_STATUS = frozenset({"ship", "partial", "wontfix_phase1"})
ALLOWED_MODULES = frozenset(
    {"pose", "face", "seg", "ocr", "audio", "depth", "zones", "clock"}
)


def load_catalog(path: Path | None = None) -> dict:
    p = path or _DEFAULT_CATALOG
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "events" not in data:
        raise ValueError(f"invalid catalog: {p}")
    return data


def load_detect_names(path: Path | None = None) -> List[str]:
    p = path or _DEFAULT_CLASSES
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return list(data["detect_names"])


def load_classes_config(path: Path | None = None) -> dict:
    p = path or _DEFAULT_CLASSES
    return yaml.safe_load(p.read_text(encoding="utf-8"))


ALLOWED_DATA_TIERS = frozenset(
    {"coco_exact", "coco_approx", "public_online", "self_collect"}
)


def validate_detect_data(classes_cfg: dict | None = None) -> None:
    """每个 detect_names 必须有 detect_data，且 data_tier 合法。"""
    cfg = classes_cfg if classes_cfg is not None else load_classes_config()
    names = list(cfg["detect_names"])
    data = cfg.get("detect_data") or {}
    missing = [n for n in names if n not in data]
    if missing:
        raise ValueError(f"detect_data missing classes: {missing}")
    extra = sorted(set(data) - set(names))
    if extra:
        raise ValueError(f"detect_data unknown classes: {extra}")
    for n, meta in data.items():
        tier = meta.get("data_tier")
        if tier not in ALLOWED_DATA_TIERS:
            raise ValueError(f"{n}: bad data_tier {tier}")
        if tier == "coco_approx" and not meta.get("coco_source"):
            raise ValueError(f"{n}: coco_approx requires coco_source")


def load_dataset_sources(path: Path | None = None) -> dict:
    p = path or _DEFAULT_DATASET_SOURCES
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "classes" not in data:
        raise ValueError(f"invalid dataset_sources: {p}")
    return data


def get_data_root(
    sources: dict | None = None,
    *,
    project_root: Path | None = None,
    override: Path | str | None = None,
) -> Path:
    """
    数据根目录（下载/自制/合并路径的前缀）。

    优先级：override（CLI）> 环境变量 PATROL_DATA_ROOT > roots.data_root > 工程根。
    相对路径相对工程根解析；绝对路径可直接指向外置硬盘。
    """
    proj = (project_root if project_root is not None else _ROOT).resolve()
    if override is not None and str(override).strip():
        p = Path(str(override)).expanduser()
        return p.resolve() if p.is_absolute() else (proj / p).resolve()

    env = (os.environ.get("PATROL_DATA_ROOT") or "").strip()
    if env:
        p = Path(env).expanduser()
        return p.resolve() if p.is_absolute() else (proj / p).resolve()
    src = sources if sources is not None else {}
    cfg = str((src.get("roots") or {}).get("data_root") or "").strip()
    if cfg:
        p = Path(cfg).expanduser()
        return p.resolve() if p.is_absolute() else (proj / p).resolve()
    return proj


def resolve_data_path(
    path: str | Path,
    sources: dict | None = None,
    *,
    project_root: Path | None = None,
    data_root: Path | None = None,
) -> Path:
    """相对路径 → data_root / path；已是绝对路径则原样 resolve。"""
    p = Path(str(path)).expanduser()
    if p.is_absolute():
        return p.resolve()
    base = (
        data_root
        if data_root is not None
        else get_data_root(sources, project_root=project_root)
    )
    return (base / p).resolve()


def iter_enabled_public(sources: dict, class_name: str) -> List[dict]:
    block = (sources.get("classes") or {}).get(class_name) or {}
    return [x for x in (block.get("public") or []) if x.get("enabled", True)]


def iter_enabled_self(sources: dict, class_name: str) -> List[dict]:
    block = (sources.get("classes") or {}).get(class_name) or {}
    return [x for x in (block.get("self") or []) if x.get("enabled", True)]


def _norm_url(url: str) -> str:
    return str(url or "").strip().rstrip("/")


def collect_public_download_jobs(
    sources: dict,
    class_names: Optional[Iterable[str]] = None,
    *,
    root: Path | None = None,
    data_root: Path | str | None = None,
) -> List[dict]:
    """
    收集 enabled public 下载任务，按 (url, local_dir) 去重。

    同一 COCO url+local_dir 被 person/bicycle/car/truck 引用时只产生一条 job。
    返回项含：url, local_dir (Path), kind, name, classes (引用该类名列表)。
    local_dir 解析：绝对路径直接用；相对路径 = data_root + 相对路径。
    """
    proj = root if root is not None else _ROOT
    base = get_data_root(sources, project_root=proj, override=data_root)
    wanted = set(class_names) if class_names is not None else None
    jobs: Dict[tuple, dict] = {}
    for cls_name, block in (sources.get("classes") or {}).items():
        if wanted is not None and cls_name not in wanted:
            continue
        for pub in block.get("public") or []:
            if not pub.get("enabled", True):
                continue
            url = _norm_url(pub.get("url") or "")
            local = str(pub.get("local_dir") or "").strip()
            if not url or not local:
                continue
            local_path = resolve_data_path(
                local, sources, project_root=proj, data_root=base
            )
            key = (url, str(local_path))
            if key not in jobs:
                kind = str(pub.get("kind") or "").strip()
                if not kind and "docs.ultralytics.com/datasets/detect/coco" in url:
                    kind = "coco"
                if not kind and "huggingface.co/datasets/" in url:
                    kind = "huggingface"
                if not kind and (
                    "tacodataset.org" in url or "github.com/pedropro/TACO" in url
                    or "github.com/pedropro/taco" in url
                ):
                    kind = "taco"
                ndjson = str(pub.get("ndjson") or "").strip() or None
                jobs[key] = {
                    "url": url,
                    "local_dir": local_path,
                    "kind": kind or "url",
                    "name": pub.get("name") or url,
                    "classes": [cls_name],
                    "ndjson": ndjson,
                    "unofficial": bool(pub.get("unofficial")),
                }
            else:
                if cls_name not in jobs[key]["classes"]:
                    jobs[key]["classes"].append(cls_name)
                if not jobs[key].get("ndjson"):
                    ndjson = str(pub.get("ndjson") or "").strip() or None
                    if ndjson:
                        jobs[key]["ndjson"] = ndjson
    return list(jobs.values())


def apply_self_dir_overrides(
    sources: dict,
    overrides: Dict[str, str],
) -> dict:
    """overrides: {class_name: dir_path}，写入该类第一条 self 并 enabled=true。"""
    out = copy.deepcopy(sources)
    for cls, directory in overrides.items():
        block = out.setdefault("classes", {}).setdefault(cls, {})
        selves = block.setdefault("self", [])
        if not selves:
            selves.append({"name": "cli-override", "dir": directory, "enabled": True})
        else:
            selves[0]["dir"] = directory
            selves[0]["enabled"] = True
    return out


def validate_dataset_sources(
    sources: dict | None = None,
    detect_names: Optional[List[str]] = None,
) -> None:
    """
    - detect_names 中每一类都应有条目
    - 若某 public 条目 enabled，则 url 非空
    - 若某 self 条目 enabled，则 dir 非空
    """
    src = sources if sources is not None else load_dataset_sources()
    names = detect_names if detect_names is not None else load_detect_names()
    classes = src.get("classes") or {}
    missing = [n for n in names if n not in classes]
    if missing:
        raise ValueError(f"dataset_sources missing classes: {missing}")
    for n in names:
        block = classes[n]
        for i, pub in enumerate(block.get("public") or []):
            if pub.get("enabled", True) and not str(pub.get("url") or "").strip():
                raise ValueError(f"{n}.public[{i}] enabled but url empty")
        for i, self_ds in enumerate(block.get("self") or []):
            if self_ds.get("enabled", True) and not str(self_ds.get("dir") or "").strip():
                raise ValueError(f"{n}.self[{i}] enabled but dir empty")


def apply_enable_overrides(
    catalog: dict,
    enable: Optional[Iterable[str]] = None,
    disable: Optional[Iterable[str]] = None,
) -> dict:
    """深拷贝后覆盖 enabled；供 CLI --enable/--disable。"""
    out = copy.deepcopy(catalog)
    en = set(enable or [])
    dis = set(disable or [])
    for e in out["events"]:
        eid = e["id"]
        if eid in en:
            e["enabled"] = True
        if eid in dis:
            e["enabled"] = False
    return out


def iter_active_events(catalog: dict) -> List[dict]:
    """使能且非一期不做的事件。"""
    active: List[dict] = []
    for e in catalog["events"]:
        if e.get("status") == "wontfix_phase1":
            continue
        if not e.get("enabled", False):
            continue
        active.append(e)
    return active


def is_event_inferable(catalog: dict, event_id: str) -> bool:
    for e in catalog["events"]:
        if e["id"] != event_id:
            continue
        if e.get("status") == "wontfix_phase1":
            return False
        return bool(e.get("enabled", False))
    return False


def detect_class_union(
    catalog: dict,
    detect_names: Optional[List[str]] = None,
) -> List[str]:
    """
    训练用检测类并集：active 事件 depends_on.detect
    ∪ train.always_include_detect，顺序对齐 classes.yaml。
    """
    names = detect_names if detect_names is not None else load_detect_names()
    order = {n: i for i, n in enumerate(names)}
    needed: Set[str] = set()
    for e in iter_active_events(catalog):
        needed.update(e.get("depends_on", {}).get("detect") or [])
    always = catalog.get("train", {}).get("always_include_detect") or []
    needed.update(always)
    unknown = needed - set(names)
    if unknown:
        raise ValueError(f"detect classes not in classes.yaml: {sorted(unknown)}")
    return sorted(needed, key=lambda n: order[n])


def modules_union(catalog: dict) -> Set[str]:
    mods: Set[str] = set()
    for e in iter_active_events(catalog):
        mods.update(e.get("depends_on", {}).get("modules") or [])
    bad = mods - ALLOWED_MODULES
    if bad:
        raise ValueError(f"unknown modules: {sorted(bad)}")
    return mods


def validate_catalog(catalog: dict, detect_names: Optional[List[str]] = None) -> None:
    names = set(detect_names if detect_names is not None else load_detect_names())
    validate_detect_data()
    validate_dataset_sources()
    for e in catalog["events"]:
        for req in ("id", "name_zh", "layer", "status", "enabled", "depends_on"):
            if req not in e:
                raise ValueError(f"event missing {req}: {e.get('id')}")
        if e["status"] not in ALLOWED_STATUS:
            raise ValueError(f"bad status for {e['id']}: {e['status']}")
        if e["status"] in ("partial", "wontfix_phase1") and not (e.get("reason") or "").strip():
            raise ValueError(f"reason required for {e['id']}")
        dep = e["depends_on"]
        for c in dep.get("detect") or []:
            if c not in names:
                raise ValueError(f"{e['id']} depends on unknown detect class {c}")
        for m in dep.get("modules") or []:
            if m not in ALLOWED_MODULES:
                raise ValueError(f"{e['id']} depends on unknown module {m}")


def summarize_enablement(catalog: dict) -> Dict[str, Any]:
    active = iter_active_events(catalog)
    return {
        "active_event_ids": [e["id"] for e in active],
        "detect_classes": detect_class_union(catalog),
        "modules": sorted(modules_union(catalog)),
        "disabled_or_wontfix": [
            e["id"]
            for e in catalog["events"]
            if not is_event_inferable(catalog, e["id"])
        ],
    }
