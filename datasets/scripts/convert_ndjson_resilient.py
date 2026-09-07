#!/usr/bin/env python3
"""稳健的 Ultralytics Platform .ndjson → YOLO 转换（可续跑、低并发、允许重试）。

官方 convert_ndjson_to_yolo 在任意图失败时抛 RuntimeError 且不写 data.yaml；
默认 Semaphore(128)+30s 超时在慢网/大图下极易中断。本脚本：
- 跳过已存在图片（续下）
- 并发可配（默认 8）
- 多轮重试缺失图
- 每轮结束后写 data.yaml（complete=是否全部成功）
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import yaml

try:
    from ultralytics.utils import clean_url
except ImportError:  # pragma: no cover

    def clean_url(url: str) -> str:  # type: ignore[misc]
        return url.split("?")[0]


IMG_OK_MIN_BYTES = 100


def _load_records(ndjson_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lines = [json.loads(line) for line in ndjson_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise SystemExit(f"ndjson 内容过少: {ndjson_path}")
    return lines[0], lines[1:]


def _assign_filenames(
    dataset_record: dict[str, Any],
    image_records: list[dict[str, Any]],
    ndjson_path: Path,
) -> str:
    """与 ultralytics converter 一致的文件名与数据集 hash（原地改 record['file']）。"""
    lines = [dataset_record, *image_records]
    _h = hashlib.sha256()
    for i, r in enumerate(lines):
        if i:
            split, source_name = r.get("split"), r.get("file")
            if split not in {"train", "val", "test"}:
                raise ValueError(f"Invalid NDJSON split: {split!r}")
            if not isinstance(source_name, str) or not source_name:
                raise ValueError(f"Invalid NDJSON image name: {source_name!r}")
            suffix = source_name.rsplit(".", 1)[-1]
            stems = (Path(clean_url(r.get("url") or "")).stem, Path(source_name).stem)
            content_hash = next(
                (s.lower() for s in stems if len(s) == 32 and all(c in "0123456789abcdef" for c in s)),
                None,
            )
            stem = f"{content_hash}_{i}" if content_hash else i
            r["file"] = f"{stem}.{suffix}" if suffix.isalnum() and len(suffix) <= 10 else f"{stem}.jpg"
        hash_record = {k: v for k, v in r.items() if k != "url"}
        if r.get("file"):
            hash_record["_source"] = clean_url(r["url"]) if r.get("url") else str(ndjson_path.parent.resolve())
        _h.update(json.dumps(hash_record, sort_keys=True).encode())
    return _h.hexdigest()[:8]


def _clear_stale_locks(output_path: Path) -> None:
    for p in output_path.glob(".*"):
        if p.suffix == ".lock" or p.name.endswith(".lock"):
            try:
                p.unlink()
                print(f"[ndjson] removed stale lock {p.name}", flush=True)
            except OSError as e:
                print(f"[ndjson] warn: cannot remove {p}: {e}", flush=True)


def _write_label(dataset_dir: Path, record: dict[str, Any]) -> None:
    split = record["split"]
    original_name = record["file"]
    stem = original_name.rsplit(".", 1)[0] or original_name
    label_path = dataset_dir / "labels" / split / f"{stem}.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    annotations = record.get("annotations", {})
    lines_to_write: list[str] = []
    for key in annotations:
        lines_to_write = [" ".join(map(str, item)) for item in annotations[key]]
        break
    label_path.write_text("\n".join(lines_to_write) + ("\n" if lines_to_write else ""), encoding="utf-8")


def _write_data_yaml(
    dataset_dir: Path,
    dataset_record: dict[str, Any],
    image_records: list[dict[str, Any]],
    *,
    ds_hash: str,
    complete: bool,
) -> Path:
    class_names = {int(k): v for k, v in dataset_record.get("class_names", {}).items()}
    splits = {r["split"] for r in image_records}
    data_yaml: dict[str, Any] = {
        k: v for k, v in dataset_record.items() if k not in ("class_names", "type")
    }
    if class_names:
        data_yaml["names"] = class_names
    for split in sorted(splits):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        data_yaml[split] = f"images/{split}"
    data_yaml["path"] = str(dataset_dir.resolve())
    data_yaml["hash"] = ds_hash
    data_yaml["complete"] = complete
    out = dataset_dir / "data.yaml"
    out.write_text(yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out


async def _download_one(
    session: Any,
    path: Path,
    url: str,
    *,
    timeout: float,
) -> bool:
    if path.is_file() and path.stat().st_size >= IMG_OK_MIN_BYTES:
        return True
    if not url:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    import aiohttp

    for attempt in range(4):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                response.raise_for_status()
                data = await response.read()
                if len(data) < IMG_OK_MIN_BYTES:
                    raise OSError(f"too small ({len(data)} bytes)")
                tmp = path.with_suffix(path.suffix + ".part")
                tmp.write_bytes(data)
                tmp.replace(path)
                return True
        except Exception as e:
            if attempt >= 3:
                print(f"[warn] fail {clean_url(url)}: {type(e).__name__}: {e}", flush=True)
                return False
            await asyncio.sleep(1.5 * (2**attempt) + random.random())
    return False


async def convert_ndjson_resilient(
    ndjson_path: Path,
    output_path: Path,
    *,
    concurrency: int = 8,
    timeout: float = 120.0,
    max_rounds: int = 8,
    allow_partial: bool = False,
) -> Path:
    """返回 data.yaml 路径。默认要求全部成功；allow_partial 时也写 complete=False 并返回。"""
    import aiohttp

    dataset_record, image_records = _load_records(ndjson_path)
    if dataset_record.get("task", "detect") != "detect":
        raise SystemExit(f"仅支持 detect，当前 task={dataset_record.get('task')}")

    # val 缺失时与官方一致：从 train 抽 10%
    splits = {r["split"] for r in image_records}
    if "val" not in splits:
        train_records = [r for r in image_records if r.get("split") == "train"]
        random.Random(0).shuffle(train_records)
        val_count = max(1, len(train_records) // 10)
        for r in train_records[:val_count]:
            r["split"] = "val"

    ds_hash = _assign_filenames(dataset_record, image_records, ndjson_path)
    dataset_dir = output_path / f"{ndjson_path.stem}-{ds_hash}"
    output_path.mkdir(parents=True, exist_ok=True)
    _clear_stale_locks(output_path)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for r in image_records:
        _write_label(dataset_dir, r)

    missing = list(image_records)
    for round_i in range(1, max_rounds + 1):
        todo = [
            r
            for r in missing
            if not (
                (p := dataset_dir / "images" / r["split"] / r["file"]).is_file()
                and p.stat().st_size >= IMG_OK_MIN_BYTES
            )
        ]
        ok_before = len(image_records) - len(todo)
        print(
            f"[ndjson] round {round_i}/{max_rounds}: have {ok_before}/{len(image_records)}, "
            f"fetch {len(todo)} (concurrency={concurrency})",
            flush=True,
        )
        if not todo:
            break

        sem = asyncio.Semaphore(concurrency)

        async def one(rec: dict[str, Any]) -> bool:
            async with sem:
                img = dataset_dir / "images" / rec["split"] / rec["file"]
                return await _download_one(session, img, rec.get("url") or "", timeout=timeout)

        connector = aiohttp.TCPConnector(limit=concurrency, ttl_dns_cache=300)
        async with aiohttp.ClientSession(trust_env=True, connector=connector) as session:
            results = await asyncio.gather(*[one(r) for r in todo])
        got = sum(1 for x in results if x)
        print(f"[ndjson] round {round_i}: fetched {got}/{len(todo)}", flush=True)
        missing = todo
        if got == 0:
            print("[ndjson] no progress this round", flush=True)
            break

    final_ok = sum(
        1
        for r in image_records
        if (p := dataset_dir / "images" / r["split"] / r["file"]).is_file()
        and p.stat().st_size >= IMG_OK_MIN_BYTES
    )
    complete = final_ok == len(image_records)
    yaml_path = _write_data_yaml(
        dataset_dir, dataset_record, image_records, ds_hash=ds_hash, complete=complete
    )
    print(
        f"[ndjson] done {final_ok}/{len(image_records)} → {yaml_path} (complete={complete})",
        flush=True,
    )
    if not complete and not allow_partial:
        raise RuntimeError(
            f"Downloaded {final_ok}/{len(image_records)} images from {ndjson_path}; "
            f"re-run to resume, or pass --allow-partial"
        )
    return yaml_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ndjson", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="输出父目录（其下会建 datasets-<hash>/）")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="即使仍有缺图也退出 0（data.yaml complete=false）",
    )
    args = ap.parse_args(argv)
    try:
        path = asyncio.run(
            convert_ndjson_resilient(
                args.ndjson.expanduser(),
                args.out.expanduser(),
                concurrency=args.concurrency,
                timeout=args.timeout,
                max_rounds=args.max_rounds,
                allow_partial=args.allow_partial,
            )
        )
    except RuntimeError as e:
        print(f"[fail] {e}", file=sys.stderr)
        return 1
    print(f"[ok] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
