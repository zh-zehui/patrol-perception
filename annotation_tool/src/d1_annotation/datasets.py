from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml
from PIL import Image

from .core import DatasetError, IMAGE_SUFFIXES, load_classes


@dataclass(frozen=True)
class DatasetSource:
    id: str
    name: str
    root: Path
    source_type: str = "self_collected"
    read_only: bool = True
    exhaustive_classes: tuple[int, ...] = ()


@dataclass(frozen=True)
class Sample:
    source_id: str
    split: str
    image: Path
    label: Path
    key: str
    class_ids: frozenset[int]
    box_count: int
    eligible: bool = True


def load_registry(path: Path) -> list[DatasetSource]:
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    result = []
    seen = set()
    for entry in raw.get("datasets", []):
        source_id = str(entry["id"])
        if source_id in seen:
            raise DatasetError(f"Duplicate dataset id: {source_id}")
        seen.add(source_id)
        result.append(DatasetSource(
            id=source_id,
            name=str(entry.get("name", source_id)),
            root=Path(entry["path"]).expanduser().resolve(),
            source_type=str(entry.get("source_type", "self_collected")),
            read_only=bool(entry.get("read_only", True)),
            exhaustive_classes=tuple(int(x) for x in entry.get("exhaustive_classes", [])),
        ))
    return result


def save_registry(path: Path, sources: list[DatasetSource]) -> None:
    payload = {"version": 1, "datasets": [
        {"id": s.id, "name": s.name, "path": str(s.root), "source_type": s.source_type,
         "read_only": s.read_only, "exhaustive_classes": list(s.exhaustive_classes)}
        for s in sources
    ]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _read_label(path: Path, class_count: int) -> tuple[list[dict[str, Any]], list[str], Counter[int], int]:
    boxes = []
    errors = []
    raw_class_counts: Counter[int] = Counter()
    raw_lines = 0
    if not path.exists():
        return boxes, ["missing_label"], raw_class_counts, raw_lines
    for line_no, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not raw.strip():
            continue
        raw_lines += 1
        fields = raw.split()
        try:
            class_id = int(float(fields[0]))
            raw_class_counts[class_id] += 1
            values = [float(x) for x in fields[1:]]
        except (ValueError, IndexError):
            errors.append(f"line_{line_no}:number")
            continue
        if class_id < 0 or class_id >= class_count:
            errors.append(f"line_{line_no}:class_id")
            continue
        # Accept YOLO segmentation polygons for statistics/building. They are
        # converted to an enclosing detection box; the source file is untouched.
        if len(values) >= 6 and len(values) % 2 == 0:
            xs, ys = values[0::2], values[1::2]
            if not all(0 <= x <= 1 for x in values) or len(xs) < 3:
                errors.append(f"line_{line_no}:polygon_coordinates")
                continue
            xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
            boxes.append({"class_id": class_id, "x": (xmin+xmax)/2, "y": (ymin+ymax)/2,
                          "width": xmax-xmin, "height": ymax-ymin, "source_format": "segmentation"})
            continue
        if len(fields) != 5:
            errors.append(f"line_{line_no}:columns")
            continue
        if not all(0 <= x <= 1 for x in values) or values[2] <= 0 or values[3] <= 0:
            errors.append(f"line_{line_no}:coordinates")
            continue
        x, y, w, h = values
        if x - w / 2 < 0 or x + w / 2 > 1 or y - h / 2 < 0 or y + h / 2 > 1:
            errors.append(f"line_{line_no}:outside")
            continue
        boxes.append({"class_id": class_id, "x": x, "y": y, "width": w, "height": h})
    return boxes, errors, raw_class_counts, raw_lines


def _scan_payload(scan: dict[str, Any]) -> dict[str, Any]:
    payload = {k: v for k, v in scan.items() if k != "samples"}
    payload["samples"] = [
        {"source_id": x.source_id, "split": x.split, "image": str(x.image), "label": str(x.label),
         "key": x.key, "class_ids": sorted(x.class_ids), "box_count": x.box_count,
         "eligible": x.eligible}
        for x in scan["samples"]
    ]
    return payload


def _scan_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["samples"] = [Sample(x["source_id"], x["split"], Path(x["image"]), Path(x["label"]),
                             x["key"], frozenset(x["class_ids"]), x["box_count"],
                             bool(x.get("eligible", True)))
                         for x in payload.get("samples", [])]
    return result


def scan_cache_path(cache_dir: Path, source_id: str) -> Path:
    return cache_dir / f"{source_id}.scan_cache.json"


def load_scan_cache(cache_dir: Path, source: DatasetSource) -> dict[str, Any] | None:
    path = scan_cache_path(cache_dir, source.id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != 4 or payload.get("source_path") != str(source.root):
            return None
        return _scan_from_payload(payload["scan"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_scan_cache(cache_dir: Path, scan: dict[str, Any]) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = scan_cache_path(cache_dir, scan["id"])
    path.write_text(json.dumps({"version": 4, "source_path": scan["path"],
                                "scanned_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                                "scan": _scan_payload(scan)}, ensure_ascii=False), encoding="utf-8")
    return path


def scan_source(source: DatasetSource, classes: list[dict[str, Any]],
                progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    root = source.root
    if not root.is_dir():
        raise DatasetError(f"Dataset path does not exist: {root}")
    samples: list[Sample] = []
    errors = Counter()
    class_images = Counter(); class_boxes = Counter(); raw_class_boxes = Counter(); size_bins = Counter()
    eligible_class_images = Counter(); eligible_class_boxes = Counter()
    split_stats: dict[str, dict[str, int]] = {}
    images_dir = root / "images"
    labels_dir = root / "labels"
    image_paths_by_split = {}
    total_images = 0
    for split in ("train", "val", "test"):
        split_images = images_dir / split
        paths = sorted(p for p in split_images.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES) if split_images.is_dir() else []
        image_paths_by_split[split] = paths
        total_images += len(paths)
    processed_images = 0
    for split in ("train", "val", "test"):
        split_images = images_dir / split
        if not split_images.is_dir():
            continue
        stats = {"images": 0, "label_files": 0, "labeled_images": 0, "eligible_images": 0,
                 "physical_empty_labels": 0, "pure_abnormal_label_files": 0,
                 "no_valid_boxes": 0, "abnormal_label_files": 0, "abnormal_boxes": 0,
                 "raw_boxes": 0, "valid_boxes": 0}
        for image in image_paths_by_split[split]:
            stats["images"] += 1
            relative = image.relative_to(images_dir)
            label = labels_dir / relative.with_suffix(".txt")
            boxes, row_errors, raw_counts, raw_lines = _read_label(label, len(classes))
            for cid, count in raw_counts.items(): raw_class_boxes[cid] += count
            for error in row_errors: errors[error] += 1
            if not label.exists():
                errors["missing_labels"] += 1
            else:
                stats["label_files"] += 1
                raw_count = sum(raw_counts.values())
                stats["raw_boxes"] += raw_lines
                if raw_lines == 0:
                    stats["physical_empty_labels"] += 1
                if row_errors:
                    stats["abnormal_label_files"] += 1
                    stats["abnormal_boxes"] += len(row_errors)
                if not boxes:
                    stats["no_valid_boxes"] += 1
                    if raw_lines > 0 and row_errors:
                        stats["pure_abnormal_label_files"] += 1
                else:
                    stats["labeled_images"] += 1
            stats["valid_boxes"] += len(boxes)
            stats["boxes"] = stats["valid_boxes"]
            ids = frozenset(b["class_id"] for b in boxes)
            for b in boxes:
                class_boxes[b["class_id"]] += 1
                area = b["width"] * b["height"]
                size_bins[(b["class_id"], "small" if area < .01 else "large" if area > .1 else "medium")] += 1
            key = image.relative_to(root).as_posix()
            # Dataset construction is conservative: a sample is eligible only
            # when it has a label file, at least one valid box, and no abnormal
            # annotation line at all. A mixed valid+invalid label is excluded.
            eligible = label.exists() and raw_lines > 0 and bool(boxes) and not row_errors
            if eligible:
                stats["eligible_images"] += 1
                for b in boxes:
                    eligible_class_boxes[b["class_id"]] += 1
                for cid in ids:
                    eligible_class_images[cid] += 1
            samples.append(Sample(source.id, split, image, label, key, ids, len(boxes), eligible))
            processed_images += 1
            if progress is not None:
                progress(processed_images, total_images)
        split_stats[split] = stats
    quality = {
        "label_files": sum(x["label_files"] for x in split_stats.values()),
        "missing_labels": errors["missing_labels"],
        "physical_empty_labels": sum(x["physical_empty_labels"] for x in split_stats.values()),
        "pure_abnormal_label_files": sum(x["pure_abnormal_label_files"] for x in split_stats.values()),
        "no_valid_boxes": sum(x["no_valid_boxes"] for x in split_stats.values()),
        "abnormal_label_files": sum(x["abnormal_label_files"] for x in split_stats.values()),
        "abnormal_boxes": sum(x["abnormal_boxes"] for x in split_stats.values()),
        "raw_boxes": sum(x["raw_boxes"] for x in split_stats.values()),
        "valid_boxes": sum(x["valid_boxes"] for x in split_stats.values()),
        "eligible_images": sum(x["eligible_images"] for x in split_stats.values()),
    }
    quality["invalid_rate"] = (quality["abnormal_boxes"] / quality["raw_boxes"]
                               if quality["raw_boxes"] else 0.0)

    # class_images was accumulated per box above; recalculate exactly from samples.
    class_images = Counter()
    for sample in samples:
        for class_id in sample.class_ids: class_images[class_id] += 1
    class_stats = {}
    for entry in classes:
        cid = int(entry["id"])
        class_stats[str(cid)] = {
            "name": entry["name"], "images": eligible_class_images[cid], "boxes": eligible_class_boxes[cid],
            "valid_images": eligible_class_images[cid], "valid_boxes": eligible_class_boxes[cid],
            "all_valid_images": class_images[cid], "all_valid_boxes": class_boxes[cid],
            "raw_boxes": raw_class_boxes[cid],
            "abnormal_boxes": raw_class_boxes[cid] - class_boxes[cid],
            "sizes": {bucket: size_bins[(cid, bucket)] for bucket in ("small", "medium", "large")},
        }
    return {
        "id": source.id, "name": source.name, "path": str(root), "source_type": source.source_type,
        "read_only": source.read_only, "images": len(samples), "boxes": sum(s.box_count for s in samples),
        "valid_boxes": sum(s.box_count for s in samples),
        "splits": split_stats, "classes": class_stats, "quality": quality,
        "errors": dict(errors), "samples": samples,
    }


def aggregate_scans(scans: list[dict[str, Any]], classes: list[dict[str, Any]]) -> dict[str, Any]:
    result = {str(c["id"]): {"name": c["name"], "images": 0, "boxes": 0,
              "raw_boxes": 0, "valid_boxes": 0, "abnormal_boxes": 0, "datasets": {}}
              for c in classes}
    quality = Counter()
    for scan in scans:
        quality.update({k: v for k, v in scan.get("quality", {}).items()
                        if k != "invalid_rate" and isinstance(v, int)})
        for cid, stats in scan["classes"].items():
            result[cid]["images"] += stats["images"]
            result[cid]["boxes"] += stats["boxes"]
            result[cid]["raw_boxes"] += stats.get("raw_boxes", stats["boxes"])
            result[cid]["valid_boxes"] += stats.get("valid_boxes", stats["boxes"])
            result[cid]["abnormal_boxes"] += stats.get("abnormal_boxes", 0)
            result[cid]["datasets"][scan["id"]] = {
                "images": stats["images"], "raw_boxes": stats.get("raw_boxes", stats["boxes"]),
                "valid_boxes": stats.get("valid_boxes", stats["boxes"]),
                "abnormal_boxes": stats.get("abnormal_boxes", 0),
            }
    quality_result = dict(quality)
    quality_result["invalid_rate"] = (quality["abnormal_boxes"] / quality["raw_boxes"]
                                      if quality["raw_boxes"] else 0.0)
    return {"datasets": [{k: v for k, v in scan.items() if k != "samples"} for scan in scans],
            "classes": result, "quality": quality_result}


def build_dataset(scans: list[dict[str, Any]], classes: list[dict[str, Any]], output: Path,
                  selected_classes: list[int], targets: dict[int, int], seed: int = 0,
                  split_ratio: tuple[float, float, float] = (.7, .2, .1),
                  source_ratios: dict[str, float] | None = None,
                  dataset_targets: dict[str, dict[int, int]] | None = None,
                  dry_run: bool = False,
                  progress: Callable[[str, int, int], None] | None = None) -> dict[str, Any]:
    selected_classes = sorted({int(cid) for cid in selected_classes})
    if not selected_classes:
        raise DatasetError("selected_classes cannot be empty")
    if any(cid < 0 or cid >= len(classes) for cid in selected_classes):
        raise DatasetError("selected_classes contains an unknown class id")
    if any(int(value) < 0 for value in targets.values()):
        raise DatasetError("targets must be non-negative")
    if len(split_ratio) != 3 or any(float(value) < 0 for value in split_ratio) or abs(sum(split_ratio) - 1.0) > 1e-6:
        raise DatasetError("split_ratio must contain three non-negative values summing to 1")
    dataset_targets = dataset_targets or {}
    source_ids = {scan["id"] for scan in scans}
    unknown_target_sources = set(dataset_targets) - source_ids
    if unknown_target_sources:
        raise DatasetError(f"dataset_targets contains unknown datasets: {sorted(unknown_target_sources)}")
    for source_id, values in dataset_targets.items():
        if any(int(cid) not in selected_classes for cid in values):
            raise DatasetError(f"dataset_targets for {source_id} contains an unselected class")
        if any(int(value) < 0 for value in values.values()):
            raise DatasetError("dataset_targets must be non-negative")
    source_types = {scan["id"]: scan.get("source_type", "unknown") for scan in scans}
    def quota(sample: Sample, cid: int) -> int:
        if dataset_targets:
            return int(dataset_targets.get(sample.source_id, {}).get(cid, 0))
        return int(targets.get(cid, 0))
    candidates = [sample for scan in scans for sample in scan["samples"]
                  if sample.eligible and sample.class_ids.intersection(selected_classes)
                  and (not dataset_targets or any(quota(sample, cid) > 0 for cid in sample.class_ids.intersection(selected_classes)))]
    if progress:
        progress("准备候选图片", len(candidates), len(candidates))
    rng = random.Random(seed)
    rng.shuffle(candidates)

    # Build a source/class inverted index once. The previous implementation
    # rescanned every candidate after each selected image (O(quota*candidates))
    # and rebuilt source counters inside that scan. Large datasets therefore
    # became progressively slower. The indexed sampler only examines candidates
    # belonging to the currently most constrained source/class quota.
    candidate_pools: dict[tuple[str, int], list[int]] = defaultdict(list)
    requested: dict[tuple[str, int], int] = {}
    global_source = "__global__"
    if dataset_targets:
        for source_id in source_ids:
            for cid in selected_classes:
                target = int(dataset_targets.get(source_id, {}).get(cid, 0))
                if target > 0:
                    requested[(source_id, cid)] = target
    else:
        for cid in selected_classes:
            target = int(targets.get(cid, 0))
            if target > 0:
                requested[(global_source, cid)] = target
    def quota_key(sample: Sample, cid: int) -> tuple[str, int]:
        return (sample.source_id, cid) if dataset_targets else (global_source, cid)
    for candidate_index, sample in enumerate(candidates):
        for cid in sample.class_ids.intersection(selected_classes):
            key = quota_key(sample, cid)
            if key in requested:
                candidate_pools[key].append(candidate_index)

    selected: list[Sample] = []
    selected_indices: set[int] = set()
    counts = Counter()
    dataset_counts = Counter()
    quota_counts = Counter()
    source_counts = Counter()
    pool_offsets = {key: 0 for key in candidate_pools}
    remaining = {key for key in requested if candidate_pools.get(key)}
    selection_total = max(1, sum(requested.values()))
    fulfilled_units = 0
    lookahead = 48

    while remaining:
        # Handle scarce quotas first. There are normally only classes*datasets
        # keys here, so recalculating this small priority list is inexpensive.
        total_selected = max(1, len(selected))
        source_type_counts = Counter(source_types.get(source_id, source_id) * 0
                                     for source_id in ())  # typed empty Counter
        for source_id, count in source_counts.items():
            source_type_counts[source_types.get(source_id, source_id)] += count

        def key_priority(key: tuple[str, int]) -> tuple[float, int, str, int]:
            deficit = requested[key] - dataset_counts[key]
            pool = candidate_pools[key]
            offset = pool_offsets.get(key, 0)
            availability_ratio = max(0, len(pool) - offset) / max(1, deficit)
            ratio_bonus = 0.0
            ratio_key = source_types.get(key[0], key[0])
            if source_ratios and ratio_key in source_ratios:
                actual = source_type_counts[ratio_key] / total_selected
                ratio_bonus = max(0.0, float(source_ratios[ratio_key]) - actual)
            return (availability_ratio - ratio_bonus, -deficit, key[0], key[1])

        key = min(remaining, key=key_priority)
        pool = candidate_pools[key]
        offset = pool_offsets.get(key, 0)
        while offset < len(pool) and pool[offset] in selected_indices:
            offset += 1
        pool_offsets[key] = offset
        if offset >= len(pool):
            remaining.discard(key)
            continue

        # A short look-ahead retains the useful multi-class preference without
        # scanning tens of thousands of unrelated candidates on every step.
        best_index = None
        best_score = (-1, -1)
        inspected = 0
        cursor = offset
        while cursor < len(pool) and inspected < lookahead:
            candidate_index = pool[cursor]
            cursor += 1
            if candidate_index in selected_indices:
                continue
            inspected += 1
            sample = candidates[candidate_index]
            useful_keys = [
                quota_key(sample, cid)
                for cid in sample.class_ids.intersection(selected_classes)
                if quota_key(sample, cid) in remaining
            ]
            useful = len(useful_keys)
            deficit_score = sum(requested[item] - dataset_counts[item]
                                for item in useful_keys)
            score = (useful, deficit_score)
            if score > best_score:
                best_index, best_score = candidate_index, score

        if best_index is None:
            remaining.discard(key)
            continue

        best = candidates[best_index]
        selected_indices.add(best_index)
        selected.append(best)
        source_counts[best.source_id] += 1
        gained = 0
        for cid in best.class_ids.intersection(selected_classes):
            item = quota_key(best, cid)
            if item not in remaining:
                continue
            counts[cid] += 1
            quota_counts[item] += 1
            if dataset_targets:
                dataset_counts[(best.source_id, cid)] += 1
            gained += 1
            if quota_counts[item] >= requested[item]:
                remaining.discard(item)
        fulfilled_units += gained
        if progress:
            progress("按类别配额选择图片", min(fulfilled_units, selection_total), selection_total)

    if progress:
        progress("按类别配额选择图片", selection_total, selection_total)
    # deterministic split by selected samples, retaining source split when possible is deferred.
    rng.shuffle(selected)
    n = len(selected); train_end = int(n * split_ratio[0]); val_end = train_end + int(n * split_ratio[1])
    splits = [("train", selected[:train_end]), ("val", selected[train_end:val_end]), ("test", selected[val_end:])]
    manifest = []; split_counts = Counter(); source_counts = Counter(x.source_id for x in selected)
    selected_by_dataset = {}
    for (source_id, cid), value in sorted(dataset_counts.items()):
        selected_by_dataset.setdefault(source_id, {})[str(cid)] = value
    requested_by_dataset = {
        source_id: {
            str(cid): int(dataset_targets.get(source_id, {}).get(cid, 0)) if dataset_targets
            else int(targets.get(cid, 0))
            for cid in selected_classes
        }
        for source_id in source_ids
    }
    shortage_by_dataset = {
        source_id: {cid: max(0, requested_by_dataset[source_id].get(str(cid), 0) - selected_by_dataset.get(source_id, {}).get(str(cid), 0))
                    for cid in selected_classes}
        for source_id in source_ids
    }
    report = {"output": str(output), "selected_classes": selected_classes, "targets": {str(k): v for k, v in targets.items()},
              "dataset_targets": dataset_targets, "eligible_candidates": len(candidates), "selected_images": len(selected),
              "selected_by_class": dict(counts), "selected_by_dataset": selected_by_dataset,
              "available_shortage": {str(cid): max(0, targets.get(cid, 0)-counts[cid]) for cid in selected_classes} if not dataset_targets else shortage_by_dataset,
              "source_counts": dict(source_counts), "source_ratios_requested": source_ratios or {}, "split_ratio": list(split_ratio), "seed": seed, "dry_run": dry_run}
    if dry_run:
        if progress:
            progress("生成 Dry Run 报告", 1, 1)
        return report
    output.mkdir(parents=True, exist_ok=True)
    copied = 0
    for split, items in splits:
        for index, sample in enumerate(items):
            safe = f"{sample.source_id}__{sample.split}__{sample.image.stem}"
            image_out = output / "images" / split / f"{safe}{sample.image.suffix.lower()}"
            label_out = output / "labels" / split / f"{safe}.txt"
            image_out.parent.mkdir(parents=True, exist_ok=True); label_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sample.image, image_out)
            lines = []
            source_boxes, _, _, _ = _read_label(sample.label, len(classes))
            for box in source_boxes:
                if box["class_id"] in selected_classes:
                    lines.append(f"{box['class_id']} {box['x']:.6f} {box['y']:.6f} {box['width']:.6f} {box['height']:.6f}")
            label_out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            split_counts[split] += 1
            manifest.append({"image": str(image_out.relative_to(output)), "label": str(label_out.relative_to(output)),
                             "source_dataset": sample.source_id, "source_image": str(sample.image), "source_split": sample.split,
                             "class_ids": sorted(sample.class_ids.intersection(selected_classes)), "split": split})
            copied += 1
            if progress:
                progress("复制图片和标签", copied, max(1, len(selected)))
    names = {int(c["id"]): c["name"] for c in classes}
    (output / "data.yaml").write_text(yaml.safe_dump({"path": str(output), "train": "images/train", "val": "images/val", "test": "images/test", "names": names}, allow_unicode=True, sort_keys=False), encoding="utf-8")
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for item in manifest: f.write(json.dumps(item, ensure_ascii=False) + "\n")
    report.update({"splits": dict(split_counts), "dry_run": False})
    (output / "statistics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if progress:
        progress("写入配置和统计报告", 1, 1)
    return report
