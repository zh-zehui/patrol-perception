#!/usr/bin/env python3
"""按登记 job 增量并入或移除 YOLO 数据集。

子命令（登记见 config/merge_jobs.yaml）：

  list
      列出 job、prefix、candidates 与 merge_meta 状态（present/absent）。
  append <job> [--keep A,B] [-y] [--dry-run]
      首次并入；交互选 keep；-y 非交互全 keep；已 present 则跳过。
  remove <job> [--drop A,B] [-y] [--dry-run]
      方案 B：整包 remove 后，若 remain 非空则再 append；-y 整包删不重 append。

全局 names 不因 remove 收缩；去掉的类 id 仍保留在 data.yaml。

示例::

  merge_job_cli.py list
  merge_job_cli.py append coco_pet
  merge_job_cli.py append coco_pet -y
  merge_job_cli.py remove coco_pet --drop pet
  merge_job_cli.py remove coco_pet -y

旧接口：append --config …；remove --prefix …
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import yaml

from merge_job_lib import (
    AppendPlan,
    SkipPlan,
    get_job,
    load_jobs_registry,
    load_merge_meta,
    plan_append,
    plan_remove,
    prefix_keep,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path("/media/sribd/C02E01372E01284C/patrol_perception")


def _csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("列表不能为空")
    return items


def _resolve_from_root(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _write_temp_yaml(data: dict) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="merge_job_", delete=False, encoding="utf-8"
    ) as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)
        return Path(handle.name)


def _merge_command(*args: str) -> list[str]:
    return [sys.executable, str(ROOT / "datasets/scripts/merge_yolo_sets.py"), *args]


def _run_remap(src: Path, out: Path, map_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "datasets/scripts/remap_yolo_set.py"),
            "--src",
            str(src),
            "--out",
            str(out),
            "--map",
            str(_resolve_from_root(map_path)),
        ],
        check=True,
    )


def _execute_append_plan(plan: AppendPlan, out: Path) -> None:
    if plan.need_remap:
        assert plan.map_path is not None
        _run_remap(plan.src, plan.remap_out, plan.map_path)
        for extra_src, extra_out in plan.extra_remaps:
            _run_remap(extra_src, extra_out, plan.map_path)

    temp_config = _write_temp_yaml(plan.merge_yaml)
    try:
        subprocess.run(
            _merge_command(
                "--mode",
                "append",
                "--config",
                str(temp_config),
                "--out",
                str(out),
            ),
            check=True,
        )
    finally:
        temp_config.unlink(missing_ok=True)


def _cmd_list(args: argparse.Namespace) -> int:
    registry = load_jobs_registry(args.jobs)
    meta = load_merge_meta(args.out) or {}
    print(f"DATA_ROOT={args.data_root}")
    print(f"OUT={args.out}")
    for name, job in registry.items():
        keep = prefix_keep(meta, job.prefix)
        state = (
            "absent"
            if keep is None
            else f"present keep=[{', '.join(keep)}]"
        )
        candidates = ", ".join(job.candidates)
        print(
            f"{name} {state} prefix={job.prefix} candidates=[{candidates}]"
        )
    return 0


def _cmd_append_legacy(args: argparse.Namespace) -> int:
    config = _resolve_from_root(args.config)
    if args.map:
        if not args.src or not args.remap_out:
            raise ValueError("有 --map 时需要 --src 与 --remap-out")
        remap = [
            sys.executable,
            str(ROOT / "datasets/scripts/remap_yolo_set.py"),
            "--src",
            str(args.src),
            "--out",
            str(args.remap_out),
            "--map",
            str(_resolve_from_root(args.map)),
        ]
        if args.dry_run:
            print("[dry-run] " + shlex.join(remap))
        else:
            subprocess.run(remap, check=True)
    command = _merge_command(
        "--mode", "append", "--config", str(config), "--out", str(args.out)
    )
    if args.dry_run:
        print("[dry-run] " + shlex.join(command))
    else:
        subprocess.run(command, check=True)
    return 0


def _cmd_append(args: argparse.Namespace) -> int:
    if args.config:
        return _cmd_append_legacy(args)
    if not args.job:
        raise ValueError("append 需要 job 名或 --config")

    registry = load_jobs_registry(args.jobs)
    job = get_job(registry, args.job)
    plan = plan_append(
        out_root=args.out,
        data_root=args.data_root,
        job=job,
        keep_arg=args.keep,
        yes=args.yes,
        input_fn=None,
    )
    if isinstance(plan, SkipPlan):
        print(f"[skip] {plan.reason}")
        return 0
    if args.dry_run:
        print(yaml.safe_dump(plan.merge_yaml, sort_keys=False, allow_unicode=True), end="")
        return 0
    _execute_append_plan(plan, args.out)
    return 0


def _cmd_remove_legacy(args: argparse.Namespace) -> int:
    command = _merge_command(
        "--mode", "remove", "--prefix", args.prefix, "--out", str(args.out)
    )
    if args.dry_run:
        print("[dry-run] " + shlex.join(command))
    else:
        subprocess.run(command, check=True)
    return 0


def _recovery_command(args: argparse.Namespace, remain: list[str]) -> str:
    command = [
        sys.executable,
        str(ROOT / "datasets/scripts/merge_job_cli.py"),
        "append",
        args.job,
        "--keep",
        ",".join(remain),
        "--out",
        str(args.out),
        "--data-root",
        str(args.data_root),
        "--jobs",
        str(args.jobs),
    ]
    return shlex.join(command)


def _cmd_remove(args: argparse.Namespace) -> int:
    if args.prefix:
        return _cmd_remove_legacy(args)
    if not args.job:
        raise ValueError("remove 需要 job 名或 --prefix")

    registry = load_jobs_registry(args.jobs)
    job = get_job(registry, args.job)
    plan = plan_remove(
        out_root=args.out,
        data_root=args.data_root,
        job=job,
        drop_arg=args.drop,
        yes=args.yes,
        input_fn=None,
    )
    if isinstance(plan, SkipPlan):
        print(f"[skip] {plan.reason}")
        return 0
    if args.dry_run:
        print(
            yaml.safe_dump(
                {
                    "prefix": plan.prefix,
                    "drop": plan.drop,
                    "remain": plan.remain,
                    "reappend": (
                        plan.reappend.merge_yaml if plan.reappend is not None else None
                    ),
                },
                sort_keys=False,
                allow_unicode=True,
            ),
            end="",
        )
        return 0

    subprocess.run(
        _merge_command(
            "--mode", "remove", "--prefix", plan.prefix, "--out", str(args.out)
        ),
        check=True,
    )
    if plan.reappend is not None:
        try:
            _execute_append_plan(plan.reappend, args.out)
        except (OSError, subprocess.CalledProcessError, yaml.YAMLError) as error:
            returncode = getattr(error, "returncode", 1) or 1
            print(
                f"[error] remove 已完成，但重新并入失败（退出码 {returncode}）。",
                file=sys.stderr,
            )
            print(
                f"[recovery] {_recovery_command(args, plan.remain)}",
                file=sys.stderr,
            )
            return returncode
    return 0


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    data_default = Path(
        os.environ.get("DATA_ROOT")
        or os.environ.get("PATROL_DATA_ROOT")
        or DEFAULT_DATA_ROOT
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=data_default)
    parser.add_argument("--jobs", type=Path, default=ROOT / "config/merge_jobs.yaml")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_parser() -> argparse.ArgumentParser:
    common = _common_parser()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "list", parents=[common], help="列出登记 job、candidates 与并入状态"
    )

    append = subparsers.add_parser(
        "append", parents=[common], help="并入登记 job"
    )
    append.add_argument("job", nargs="?")
    append.add_argument("--keep", type=_csv)
    append.add_argument("-y", "--yes", action="store_true")
    append.add_argument("--src", type=Path, help=argparse.SUPPRESS)
    append.add_argument("--map", help=argparse.SUPPRESS)
    append.add_argument("--remap-out", type=Path, help=argparse.SUPPRESS)
    append.add_argument("--config", help="旧接口：直接使用 merge 配置")

    remove = subparsers.add_parser(
        "remove", parents=[common], help="移除 job 的全部或部分类"
    )
    remove.add_argument("job", nargs="?")
    remove.add_argument("--drop", type=_csv)
    remove.add_argument("-y", "--yes", action="store_true")
    remove.add_argument("--prefix", help="旧接口：直接按 prefix 移除")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.out is None:
        args.out = args.data_root / "datasets/detect_w1_fire"
    try:
        if args.command == "list":
            return _cmd_list(args)
        if args.command == "append":
            return _cmd_append(args)
        return _cmd_remove(args)
    except KeyError as error:
        print(f"[error] 未知 job: {error.args[0]}（用 list 查看）", file=sys.stderr)
        return 2
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"[error] {error}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as error:
        print(f"[error] 子命令失败，退出码 {error.returncode}", file=sys.stderr)
        return error.returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())
