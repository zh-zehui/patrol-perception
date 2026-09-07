from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class DatasetError(ValueError):
    pass


@dataclass(frozen=True)
class Item:
    image: Path
    label: Path
    key: str


def load_classes(path: Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    classes = raw.get("classes")
    if not isinstance(classes, list) or not classes:
        raise DatasetError(f"Invalid class config: {path}")
    ids = [entry.get("id") for entry in classes]
    if ids != list(range(len(classes))):
        raise DatasetError("Class IDs must be continuous and ordered from zero")
    if len({entry.get("name") for entry in classes}) != len(classes):
        raise DatasetError("Class names must be unique")
    return classes


def _label_path(root: Path, image: Path) -> Path:
    rel = image.relative_to(root)
    parts = list(rel.parts)
    try:
        image_index = parts.index("images")
    except ValueError:
        # Flat image directories remain supported: labels live beside a sibling
        # labels directory, without modifying the source image.
        return root / "labels" / rel.with_suffix(".txt")
    parts[image_index] = "labels"
    return root.joinpath(*parts).with_suffix(".txt")


class YoloDataset:
    def __init__(self, root: Path, classes: list[dict[str, Any]], rescan: bool = False) -> None:
        self.root = root.expanduser().resolve()
        if not self.root.is_dir():
            raise DatasetError(f"Dataset directory does not exist: {self.root}")
        self.classes = classes
        self.meta_dir = self.root / ".d1_annotations"
        self.history_dir = self.meta_dir / "history"
        self.review_path = self.meta_dir / "review.json"
        self.candidates_dir = self.meta_dir / "candidates"
        self.candidate_history_dir = self.meta_dir / "candidate_history"
        self.drafts_dir = self.meta_dir / "drafts"
        self.draft_history_dir = self.meta_dir / "draft_history"
        self.commit_log_path = self.meta_dir / "commit_log.jsonl"
        self._lock = threading.RLock()
        self.image_index_path = self.meta_dir / "image_index.json"
        self._image_sizes: dict[str, tuple[int, int]] = {}
        cached_items = None if rescan else self._load_image_index()
        if cached_items is None:
            cached_items = self._scan_images()
            self._save_image_index(cached_items)
        self.items = [Item(image=image, label=_label_path(self.root, image), key=key)
                      for key, image in cached_items]
        if not self.items:
            raise DatasetError(f"No supported images found under {self.root}")
        self._review = self._load_review()

    def _scan_images(self) -> list[tuple[str, Path]]:
        images = sorted(
            p for p in self.root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            and ".d1_annotations" not in p.parts
            and "previews" not in p.parts
        )
        # A normal YOLO root has images/. Restrict to it when present, so assets
        # elsewhere under the dataset are not accidentally treated as samples.
        images_dir = self.root / "images"
        if images_dir.is_dir():
            images = [p for p in images if images_dir in p.parents]
        return [(image.relative_to(self.root).as_posix(), image) for image in images]

    def _image_tree_signature(self) -> dict[str, int | None]:
        """Cheap invalidation check; explicit --rescan remains available for deep changes."""
        images_dir = self.root / "images"
        paths = [images_dir, *(images_dir / split for split in ("train", "val", "test"))]
        result = {}
        for path in paths:
            try:
                result[str(path.relative_to(self.root))] = path.stat().st_mtime_ns
            except OSError:
                result[str(path.relative_to(self.root))] = None
        return result

    def _load_image_index(self) -> list[tuple[str, Path]] | None:
        try:
            payload = json.loads(self.image_index_path.read_text(encoding="utf-8"))
            if payload.get("version") != 1 or payload.get("dataset_root") != str(self.root):
                return None
            if payload.get("tree_signature") != self._image_tree_signature():
                return None
            result = []
            for entry in payload["images"]:
                key = str(entry["key"])
                image = self.root / key
                result.append((key, image))
                if entry.get("width") is not None and entry.get("height") is not None:
                    self._image_sizes[key] = (int(entry["width"]), int(entry["height"]))
            return result
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

    def _save_image_index(self, items: list[tuple[str, Path]]) -> None:
        payload = {
            "version": 1,
            "dataset_root": str(self.root),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tree_signature": self._image_tree_signature(),
            "images": [],
        }
        for key, _image in items:
            width, height = self._image_sizes.get(key, (None, None))
            payload["images"].append({
                "index": len(payload["images"]), "key": key,
                "width": width, "height": height,
            })
        self._atomic_write(self.image_index_path, json.dumps(payload, ensure_ascii=False) + "\n")

    def rescan_images(self) -> int:
        with self._lock:
            items = self._scan_images()
            self._save_image_index(items)
            self.items = [Item(image=image, label=_label_path(self.root, image), key=key)
                          for key, image in items]
            return len(self.items)

    def _load_review(self) -> dict[str, str]:
        if not self.review_path.exists():
            return {}
        try:
            value = json.loads(self.review_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def item(self, index: int) -> Item:
        if index < 0 or index >= len(self.items):
            raise IndexError(index)
        return self.items[index]

    def summary(self) -> dict[str, Any]:
        reviewed = sum(status == "reviewed" for status in self._review.values())
        return {"count": len(self.items), "reviewed": reviewed, "dataset": str(self.root)}

    def batch_indices(self, scope: str) -> list[int]:
        if scope == "all":
            return list(range(len(self.items)))
        if scope == "unreviewed":
            return [i for i, item in enumerate(self.items) if self._review.get(item.key) != "reviewed"]
        if scope == "unlabeled":
            return [i for i, item in enumerate(self.items) if not item.label.exists()]
        raise DatasetError(f"Unknown batch scope: {scope}")

    def image_info(self, index: int) -> dict[str, Any]:
        item = self.item(index)
        size = self._image_sizes.get(item.key)
        if size is None:
            with Image.open(item.image) as image:
                size = image.size
            self._image_sizes[item.key] = size
        width, height = size
        return {
            "index": index,
            "key": item.key,
            "name": item.image.name,
            "width": width,
            "height": height,
            "status": self._review.get(item.key, "pending"),
            "has_label": item.label.exists(),
        }

    def list_images(self) -> list[dict[str, Any]]:
        # The sidebar only needs identity and status. Do not open/stat every image
        # here: on a 100k+ dataset that would negate the persistent index cache.
        return [
            {"index": index, "key": item.key, "name": item.image.name,
             "status": self._review.get(item.key, "pending")}
            for index, item in enumerate(self.items)
        ]

    def image_page(self, page: int = 1, page_size: int = 200, query: str = "",
                   status: str = "") -> dict[str, Any]:
        if page < 1:
            raise DatasetError("page must be at least 1")
        if page_size < 1 or page_size > 500:
            raise DatasetError("page_size must be between 1 and 500")
        query = query.strip().casefold()
        valid_statuses = {"", "pending", "draft_saved", "draft_ready", "reviewed", "needs_review"}
        if status not in valid_statuses:
            raise DatasetError(f"Invalid review status filter: {status}")
        matched: list[tuple[int, Item, str]] = []
        status_counts: dict[str, int] = {
            "pending": 0, "draft_saved": 0, "draft_ready": 0,
            "reviewed": 0, "needs_review": 0,
        }
        for index, item in enumerate(self.items):
            item_status = self._review.get(item.key, "pending")
            status_counts[item_status] = status_counts.get(item_status, 0) + 1
            if query and query not in item.key.casefold():
                continue
            if status and item_status != status:
                continue
            matched.append((index, item, item_status))
        total = len(matched)
        pages = max(1, (total + page_size - 1) // page_size)
        page = min(page, pages)
        start = (page - 1) * page_size
        rows = [
            {"index": index, "key": item.key, "name": item.image.name, "status": item_status}
            for index, item, item_status in matched[start:start + page_size]
        ]
        return {
            "items": rows, "page": page, "page_size": page_size,
            "pages": pages, "total": total, "dataset_total": len(self.items),
            "status_counts": status_counts,
        }

    def read_boxes(self, index: int) -> list[dict[str, Any]]:
        return self._read_boxes_path(self.item(index).label)

    def _read_boxes_path(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        boxes: list[dict[str, Any]] = []
        for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 5:
                raise DatasetError(f"{path}:{line_no}: expected 5 columns")
            try:
                class_id = int(fields[0])
                x, y, width, height = map(float, fields[1:])
            except ValueError as exc:
                raise DatasetError(f"{path}:{line_no}: invalid number") from exc
            self._validate_box(class_id, x, y, width, height, path, line_no)
            boxes.append({"class_id": class_id, "x": x, "y": y, "width": width, "height": height})
        return boxes

    def _validate_box(self, class_id: int, x: float, y: float, width: float, height: float,
                      source: Path | None = None, line_no: int | None = None) -> None:
        location = f"{source}:{line_no}: " if source else ""
        if class_id < 0 or class_id >= len(self.classes):
            raise DatasetError(f"{location}class_id {class_id} is outside 0..{len(self.classes)-1}")
        if not all(0 <= value <= 1 for value in (x, y, width, height)):
            raise DatasetError(f"{location}coordinates must be in [0, 1]")
        if width <= 0 or height <= 0:
            raise DatasetError(f"{location}box width and height must be positive")
        eps = 1e-6
        if x - width / 2 < -eps or x + width / 2 > 1 + eps or y - height / 2 < -eps or y + height / 2 > 1 + eps:
            raise DatasetError(f"{location}box extends outside image")

    def save_boxes(self, index: int, boxes: list[dict[str, Any]]) -> None:
        item = self.item(index)
        normalized = []
        for box in boxes:
            try:
                values = (
                    int(box["class_id"]), float(box["x"]), float(box["y"]),
                    float(box["width"]), float(box["height"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise DatasetError("Each box needs class_id, x, y, width and height") from exc
            self._validate_box(*values)
            normalized.append(values)
        text = "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in normalized)
        with self._lock:
            item.label.parent.mkdir(parents=True, exist_ok=True)
            if item.label.exists():
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
                relative = item.label.relative_to(self.root)
                backup = self.history_dir / relative.parent / relative.stem / f"{stamp}.txt"
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item.label, backup)
            self._atomic_write(item.label, text)

    def history_versions(self, index: int) -> list[dict[str, Any]]:
        item = self.item(index)
        relative = item.label.relative_to(self.root)
        directory = self.history_dir / relative.parent / relative.stem
        if not directory.is_dir():
            return []
        versions = sorted(directory.glob("*.txt"), reverse=True)
        return [
            {"index": i, "name": path.stem, "boxes": len(self._read_boxes_path(path))}
            for i, path in enumerate(versions)
        ]

    def read_history(self, index: int, version_index: int) -> list[dict[str, Any]]:
        item = self.item(index)
        relative = item.label.relative_to(self.root)
        directory = self.history_dir / relative.parent / relative.stem
        versions = sorted(directory.glob("*.txt"), reverse=True) if directory.is_dir() else []
        if version_index < 0 or version_index >= len(versions):
            raise IndexError(version_index)
        return self._read_boxes_path(versions[version_index])

    def candidate_path(self, index: int) -> Path:
        item = self.item(index)
        relative = Path(item.key)
        return (self.candidates_dir / relative).with_suffix(".json")

    def read_candidates(self, index: int) -> dict[str, Any] | None:
        path = self.candidate_path(index)
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except (json.JSONDecodeError, OSError):
            return None

    def save_candidates(self, index: int, boxes: list[dict[str, Any]], provider: str, model: str,
                        overwrite: bool = True) -> bool:
        # Candidate files are deliberately separate from official YOLO labels.
        item = self.item(index)
        payload = {
            "image": item.key,
            "provider": provider,
            "model": model,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "boxes": boxes,
        }
        path = self.candidate_path(index)
        with self._lock:
            if path.exists() and not overwrite:
                return False
            if path.exists():
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
                relative = path.relative_to(self.candidates_dir)
                backup = self.candidate_history_dir / relative.parent / relative.stem / f"{stamp}.json"
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, backup)
            self._atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return True

    def clear_candidates(self, index: int) -> None:
        with self._lock:
            self.candidate_path(index).unlink(missing_ok=True)

    def draft_path(self, index: int) -> Path:
        return (self.drafts_dir / Path(self.item(index).key)).with_suffix(".json")

    def save_draft(self, index: int, boxes: list[dict[str, Any]]) -> None:
        payload = {
            "image": self.item(index).key,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "boxes": boxes,
        }
        self._atomic_write(self.draft_path(index), json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    def read_draft(self, index: int) -> dict[str, Any] | None:
        path = self.draft_path(index)
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except (json.JSONDecodeError, OSError):
            return None

    def clear_draft(self, index: int) -> None:
        self.draft_path(index).unlink(missing_ok=True)

    def commit_preview(self, indices: list[int]) -> dict[str, Any]:
        rows = []
        for index in indices:
            draft = self.read_draft(index)
            if not draft or not isinstance(draft.get("boxes"), list):
                continue
            item = self.item(index)
            rows.append({
                "index": index,
                "image": item.key,
                "has_existing_label": item.label.exists(),
                "boxes": len(draft["boxes"]),
            })
        return {"count": len(rows), "existing_labels": sum(r["has_existing_label"] for r in rows), "items": rows}

    def commit_drafts(self, indices: list[int], operator: str = "human") -> dict[str, Any]:
        preview = self.commit_preview(indices)
        committed = []
        for row in preview["items"]:
            index = row["index"]
            draft = self.read_draft(index)
            self.save_boxes(index, draft["boxes"])
            self.clear_draft(index)
            self.set_review(index, "reviewed")
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "operator": operator,
                "image": row["image"],
                "index": index,
                "boxes": row["boxes"],
                "had_existing_label": row["has_existing_label"],
                "action": "commit_draft",
            }
            self._atomic_append_jsonl(self.commit_log_path, entry)
            committed.append(row)
        return {"committed": len(committed), "items": committed}

    @staticmethod
    def _atomic_append_jsonl(path: Path, entry: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def create_drafts(self, indices: list[int], source: str = "official",
                      overwrite: bool = False,
                      progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
        """Create draft sidecars from official labels, candidates, or empty boxes."""
        if source not in {"official", "candidate", "empty"}:
            raise DatasetError(f"Invalid draft source: {source}")
        unique = sorted(set(indices))
        created = 0
        overwritten = 0
        skipped = 0
        written = []
        total = len(unique)
        for current, index in enumerate(unique, 1):
            self.item(index)
            existing = self.read_draft(index)
            if existing is not None and not overwrite:
                skipped += 1
            else:
                if source == "official":
                    boxes = self.read_boxes(index)
                elif source == "candidate":
                    candidate = self.read_candidates(index)
                    candidate_boxes = (candidate or {}).get("boxes", []) if candidate else []
                    boxes = [
                        {key: box[key] for key in ("class_id", "x", "y", "width", "height")}
                        for box in candidate_boxes
                    ]
                else:
                    boxes = []
                if existing is not None:
                    path = self.draft_path(index)
                    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
                    relative = path.relative_to(self.drafts_dir)
                    backup = self.draft_history_dir / relative.parent / relative.stem / f"{stamp}.json"
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, backup)
                    overwritten += 1
                self.save_draft(index, boxes)
                created += 1
                written.append(index)
            if progress:
                progress(current, total)
        if created:
            pending = [i for i in unique if self._review.get(self.item(i).key, "pending") == "pending"
                       and self.read_draft(i) is not None]
            if pending:
                self.set_reviews(pending, "draft_saved")
        return {"created": created, "overwritten": overwritten, "skipped": skipped,
                "source": source, "indices": unique, "written_indices": written}

    def set_reviews(self, indices: list[int], status: str,
                    progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
        """Set many review states and persist the review map only once."""
        if status not in {"pending", "draft_saved", "draft_ready", "reviewed", "needs_review"}:
            raise DatasetError(f"Invalid review status: {status}")
        unique = sorted(set(indices))
        items = [self.item(index) for index in unique]
        total = len(items)
        with self._lock:
            for current, item in enumerate(items, 1):
                self._review[item.key] = status
                if progress:
                    progress(current, total)
            self.meta_dir.mkdir(parents=True, exist_ok=True)
            self._atomic_write(self.review_path, json.dumps(self._review, ensure_ascii=False, indent=2) + "\n")
        return {"updated": total, "status": status, "indices": unique}

    def set_review(self, index: int, status: str) -> None:
        if status not in {"pending", "draft_saved", "draft_ready", "reviewed", "needs_review"}:
            raise DatasetError(f"Invalid review status: {status}")
        item = self.item(index)
        with self._lock:
            self._review[item.key] = status
            self.meta_dir.mkdir(parents=True, exist_ok=True)
            self._atomic_write(self.review_path, json.dumps(self._review, ensure_ascii=False, indent=2) + "\n")

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        except BaseException:
            Path(temp_name).unlink(missing_ok=True)
            raise
