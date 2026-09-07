from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable


class JobManager:
    def __init__(self, max_workers: int = 1) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="d1-autolabel")
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def submit(self, indices: list[int], worker: Callable[[int], int]) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        job = {
            "id": job_id, "status": "queued", "total": len(indices), "completed": 0,
            "failed": 0, "boxes": 0, "errors": [], "current": None,
            "cancel_requested": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            self._jobs[job_id] = job
        self._executor.submit(self._run, job_id, indices, worker)
        return dict(job)

    def _run(self, job_id: str, indices: list[int], worker: Callable[[int], int]) -> None:
        self._set(job_id, status="running")
        for index in indices:
            with self._lock:
                if self._jobs[job_id].get("cancel_requested"):
                    self._jobs[job_id].update(status="cancelled", finished_at=datetime.now(timezone.utc).isoformat())
                    return
            self._set(job_id, current=index)
            try:
                count = worker(index)
                self._increment(job_id, completed=1, boxes=count)
            except Exception as exc:
                self._increment(job_id, failed=1)
                with self._lock:
                    self._jobs[job_id]["errors"].append({"index": index, "error": type(exc).__name__})
            self._set(job_id, current=index)
        with self._lock:
            job = self._jobs[job_id]
            if job.get("cancel_requested"):
                job.update(status="cancelled", finished_at=datetime.now(timezone.utc).isoformat())
                return
            failed = job["failed"]
        self._set(job_id, status="completed_with_errors" if failed else "completed",
                  finished_at=datetime.now(timezone.utc).isoformat())

    def _set(self, job_id: str, **values: Any) -> None:
        with self._lock:
            self._jobs[job_id].update(values)

    def _increment(self, job_id: str, **values: int) -> None:
        with self._lock:
            for key, value in values.items():
                self._jobs[job_id][key] += value

    def cancel(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job["status"] in {"queued", "running"}:
                job["cancel_requested"] = True
                if job["status"] == "queued":
                    job["status"] = "cancelling"
            return dict(job)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None


class ReviewJobManager:
    """Background bulk review-state jobs with item progress."""
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="d1-review")
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def submit(self, indices: list[int], action: str,
               worker: Callable[[Callable[[int, int], None]], dict[str, Any]]) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        job = {"id": job_id, "status": "queued", "action": action,
               "current": 0, "total": len(indices), "percent": 0.0,
               "result": None, "error": None,
               "created_at": datetime.now(timezone.utc).isoformat()}
        with self._lock:
            self._jobs[job_id] = job
        self._executor.submit(self._run, job_id, worker)
        return dict(job)

    def _run(self, job_id: str,
             worker: Callable[[Callable[[int, int], None]], dict[str, Any]]) -> None:
        self._set(job_id, status="running")
        def report(current: int, total: int) -> None:
            self._set(job_id, current=current, total=total,
                      percent=round(current * 100 / total, 1) if total else 100.0)
        try:
            result = worker(report)
            self._set(job_id, status="completed", current=result.get("updated", result.get("created", 0)),
                      percent=100.0, result=result,
                      finished_at=datetime.now(timezone.utc).isoformat())
        except Exception as exc:
            self._set(job_id, status="failed", error=str(exc), error_type=type(exc).__name__,
                      finished_at=datetime.now(timezone.utc).isoformat())

    def _set(self, job_id: str, **values: Any) -> None:
        with self._lock:
            self._jobs[job_id].update(values)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None


class ScanCancelled(RuntimeError):
    pass


class ScanJobManager:
    """Single-worker scan jobs with thread-safe progress snapshots."""
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="d1-scan")
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def submit(self, sources: list[Any], worker: Callable[[Any, int, int, str], Any]) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        total = len(sources)
        job = {"id": job_id, "status": "queued", "datasets_total": total,
               "datasets_completed": 0, "dataset_index": 0, "dataset_id": None,
               "current": 0, "total": 0, "percent": 0, "errors": [],
               "cancel_requested": False,
               "created_at": datetime.now(timezone.utc).isoformat()}
        with self._lock: self._jobs[job_id] = job
        self._executor.submit(self._run, job_id, sources, worker)
        return dict(job)

    def _run(self, job_id: str, sources: list[Any], worker: Callable[[Any, int, int, str], Any]) -> None:
        self._set(job_id, status="running")
        for index, source in enumerate(sources):
            self._set(job_id, dataset_index=index + 1, dataset_id=source.id, current=0, total=0, percent=0)
            def report(current: int, total: int) -> None:
                with self._lock:
                    if self._jobs[job_id].get("cancel_requested"):
                        raise ScanCancelled("用户停止扫描")
                self._set(job_id, current=current, total=total,
                          percent=round(current * 100 / total, 1) if total else 100)
            try:
                worker(source, index, len(sources), report)
            except ScanCancelled:
                self._set(job_id, status="cancelled", finished_at=datetime.now(timezone.utc).isoformat())
                return
            except Exception as exc:
                with self._lock: self._jobs[job_id]["errors"].append({"dataset": source.id, "error": type(exc).__name__, "message": str(exc)})
            self._set(job_id, datasets_completed=index + 1, percent=100)
        self._set(job_id, status="completed", finished_at=datetime.now(timezone.utc).isoformat())

    def cancel(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job["status"] in {"queued", "running"}:
                job["cancel_requested"] = True
            return dict(job)

    def _set(self, job_id: str, **values: Any) -> None:
        with self._lock: self._jobs[job_id].update(values)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None


class BuildJobManager:
    """Background dataset build jobs with stage and item progress."""
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="d1-build")
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def submit(self, worker: Callable[[Callable[[str, int, int], None]], dict[str, Any]],
               dry_run: bool) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        job = {"id": job_id, "status": "queued", "stage": "等待执行",
               "current": 0, "total": 0, "percent": 0.0, "dry_run": dry_run,
               "result": None, "error": None,
               "created_at": datetime.now(timezone.utc).isoformat()}
        with self._lock:
            self._jobs[job_id] = job
        self._executor.submit(self._run, job_id, worker)
        return dict(job)

    def _run(self, job_id: str,
             worker: Callable[[Callable[[str, int, int], None]], dict[str, Any]]) -> None:
        self._set(job_id, status="running")
        def report(stage: str, current: int, total: int) -> None:
            percent = round(current * 100 / total, 1) if total else 0.0
            self._set(job_id, stage=stage, current=current, total=total, percent=percent)
        try:
            result = worker(report)
            self._set(job_id, status="completed", stage="完成", percent=100.0,
                      result=result, finished_at=datetime.now(timezone.utc).isoformat())
        except Exception as exc:
            self._set(job_id, status="failed", stage="失败", error=str(exc),
                      error_type=type(exc).__name__,
                      finished_at=datetime.now(timezone.utc).isoformat())

    def _set(self, job_id: str, **values: Any) -> None:
        with self._lock:
            self._jobs[job_id].update(values)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None
