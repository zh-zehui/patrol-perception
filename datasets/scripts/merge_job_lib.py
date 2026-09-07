"""merge_job 登记表与 append/remove 规划辅助（Task 1：加载/lookup）。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml


@dataclass
class JobSpec:
    name: str
    prefix: str
    candidates: list[str]
    src_rel: str
    remap_out_rel: str
    map_rel: str | None
    extra_roots_rel: list[str]
    extra_remap_out_rel: list[str]
    split_ratios: list[float] | None
    split_seed: int


@dataclass
class SkipPlan:
    reason: str


@dataclass
class AppendPlan:
    names: list[str]
    keep: list[str]
    merge_yaml: dict
    need_remap: bool
    src: Path
    remap_out: Path
    map_path: Path | None
    extra_remaps: list[tuple[Path, Path]] = field(default_factory=list)


@dataclass
class RemovePlan:
    prefix: str
    drop: list[str]
    remain: list[str]
    reappend: AppendPlan | None


def load_jobs_registry(path: Path) -> dict[str, JobSpec]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    jobs = raw.get("jobs") or {}
    registry: dict[str, JobSpec] = {}
    for name, spec in jobs.items():
        ratios = spec.get("split_ratios")
        registry[name] = JobSpec(
            name=name,
            prefix=spec["prefix"],
            candidates=list(spec["candidates"]),
            src_rel=spec["src_rel"],
            remap_out_rel=spec["remap_out_rel"],
            map_rel=spec.get("map_rel"),
            extra_roots_rel=list(spec.get("extra_roots_rel") or []),
            extra_remap_out_rel=list(spec.get("extra_remap_out_rel") or []),
            split_ratios=(
                [float(x) for x in ratios] if ratios is not None else None
            ),
            split_seed=int(spec.get("split_seed", 0)),
        )
    return registry


def get_job(registry: dict[str, JobSpec], name: str) -> JobSpec:
    if name not in registry:
        raise KeyError(name)
    return registry[name]


def load_merge_meta(out_root: Path) -> dict | None:
    path = Path(out_root) / "merge_meta.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def prefix_keep(meta: dict, prefix: str) -> list[str] | None:
    for s in meta.get("sets") or []:
        if s.get("prefix") == prefix:
            keep = s.get("keep")
            if not isinstance(keep, list):
                raise ValueError(
                    f"prefix {prefix!r} has missing or invalid keep in merge_meta"
                )
            return list(keep)
    return None


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
    input_fn: Callable[[str], str] | None = None,
    auto_single: bool = True,
) -> list[str]:
    """打印编号菜单，循环直到合法选择。"""
    if not opts:
        return []
    if auto_single and len(opts) == 1:
        print(f"{title}\n  仅一项，自动选择: {opts[0]}")
        return list(opts)
    read = input_fn or input
    while True:
        print(title)
        for i, o in enumerate(opts, 1):
            print(f"  [{i}] {o}")
        print("  [a] all（全部）")
        try:
            choice = read("选择编号（逗号分隔）或 a/all: ").strip()
            return parse_menu_choice(choice, opts)
        except ValueError as e:
            print(f"  无效输入: {e}，请重试")


def resolve_keep(
    *,
    candidates: list[str],
    keep_arg: list[str] | None,
    yes: bool,
    input_fn: Callable[[str], str] | None,
) -> list[str]:
    if keep_arg is not None:
        bad = [c for c in keep_arg if c not in candidates]
        if bad:
            raise ValueError(f"keep not in candidates: {bad}")
        return list(keep_arg)
    if yes:
        return list(candidates)
    return prompt_menu(candidates, title="选择要并入的类", input_fn=input_fn)


def resolve_drop(
    *,
    current: list[str],
    drop_arg: list[str] | None,
    yes: bool,
    input_fn: Callable[[str], str] | None,
) -> list[str]:
    if drop_arg is not None:
        bad = [c for c in drop_arg if c not in current]
        if bad:
            raise ValueError(f"drop not in current keep: {bad}")
        return list(drop_arg)
    if yes:
        return list(current)
    return prompt_menu(
        current,
        title="要移除的类",
        input_fn=input_fn,
        auto_single=False,
    )


def extend_names(base: list[str], keep: list[str], candidates: list[str]) -> list[str]:
    out = list(base)
    have = set(base)
    want = set(keep)
    for c in candidates:
        if c in want and c not in have:
            out.append(c)
            have.add(c)
    return out


def _load_out_names(data_yaml: Path) -> list[str]:
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = cfg.get("names")
    if isinstance(names, dict):
        return [
            str(value)
            for _, value in sorted(names.items(), key=lambda item: int(item[0]))
        ]
    if isinstance(names, list):
        return [str(name) for name in names]
    raise ValueError(f"bad names in {data_yaml}")


def build_merge_yaml_dict(
    *, names: list[str], job: JobSpec, data_root: Path, keep: list[str]
) -> dict:
    merge_set: dict = {
        "root": str(Path(data_root) / job.remap_out_rel),
        "prefix": job.prefix,
        "keep": list(keep),
    }
    # 有 split_ratios 则重划分；为 null 时沿用源集 train/val/test
    if job.split_ratios is not None:
        merge_set["split_ratios"] = list(job.split_ratios)
        merge_set["split_seed"] = job.split_seed
    if job.extra_roots_rel:
        extra_rel = (
            list(job.extra_remap_out_rel)
            if job.map_rel and job.extra_remap_out_rel
            else list(job.extra_roots_rel)
        )
        merge_set["extra_roots"] = [
            str(Path(data_root) / rel) for rel in extra_rel
        ]
    return {"version": 1, "names": list(names), "sets": [merge_set]}


def _append_plan(
    *,
    names: list[str],
    data_root: Path,
    job: JobSpec,
    keep: list[str],
) -> AppendPlan:
    data_root = Path(data_root)
    extra_remaps: list[tuple[Path, Path]] = []
    if job.map_rel and job.extra_roots_rel and job.extra_remap_out_rel:
        if len(job.extra_roots_rel) != len(job.extra_remap_out_rel):
            raise ValueError(
                f"{job.name}: extra_roots_rel and extra_remap_out_rel length mismatch"
            )
        extra_remaps = [
            (data_root / src_rel, data_root / out_rel)
            for src_rel, out_rel in zip(
                job.extra_roots_rel, job.extra_remap_out_rel
            )
        ]
    return AppendPlan(
        names=list(names),
        keep=list(keep),
        merge_yaml=build_merge_yaml_dict(
            names=names, job=job, data_root=data_root, keep=keep
        ),
        need_remap=bool(job.map_rel),
        src=data_root / job.src_rel,
        remap_out=data_root / job.remap_out_rel,
        map_path=Path(job.map_rel) if job.map_rel else None,
        extra_remaps=extra_remaps,
    )


def plan_append(
    *,
    out_root: Path,
    data_root: Path,
    job: JobSpec,
    keep_arg: list[str] | None,
    yes: bool,
    input_fn: Callable[[str], str] | None,
) -> AppendPlan | SkipPlan:
    meta = load_merge_meta(Path(out_root)) or {}
    if prefix_keep(meta, job.prefix) is not None:
        return SkipPlan(reason=f"prefix {job.prefix!r} already present")

    keep = resolve_keep(
        candidates=job.candidates,
        keep_arg=keep_arg,
        yes=yes,
        input_fn=input_fn,
    )
    old_names = _load_out_names(Path(out_root) / "data.yaml")
    names = extend_names(old_names, keep, job.candidates)
    return _append_plan(names=names, data_root=data_root, job=job, keep=keep)


def plan_remove(
    *,
    out_root: Path,
    data_root: Path,
    job: JobSpec,
    drop_arg: list[str] | None,
    yes: bool,
    input_fn: Callable[[str], str] | None,
) -> RemovePlan | SkipPlan:
    meta = load_merge_meta(Path(out_root)) or {}
    current = prefix_keep(meta, job.prefix)
    if current is None:
        return SkipPlan(reason=f"prefix {job.prefix!r} not present")

    drop = resolve_drop(
        current=current,
        drop_arg=drop_arg,
        yes=yes,
        input_fn=input_fn,
    )
    dropped = set(drop)
    remain = [name for name in current if name not in dropped]
    reappend = None
    if remain:
        names = _load_out_names(Path(out_root) / "data.yaml")
        reappend = _append_plan(
            names=names, data_root=data_root, job=job, keep=remain
        )
    return RemovePlan(
        prefix=job.prefix,
        drop=drop,
        remain=remain,
        reappend=reappend,
    )
