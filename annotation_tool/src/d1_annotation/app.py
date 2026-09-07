from __future__ import annotations

from pathlib import Path
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .autolabel import (AutoLabelError, AutoLabelProvider, create_provider,
                        provider_catalog)
from .core import DatasetError, YoloDataset, load_classes
from .jobs import BuildJobManager, JobManager, ReviewJobManager, ScanJobManager
from .datasets import (DatasetSource, aggregate_scans, build_dataset, load_registry, save_registry,
                       scan_source, load_scan_cache, save_scan_cache)

STATIC_DIR = Path(__file__).with_name("static")


class Box(BaseModel):
    class_id: int
    x: float
    y: float
    width: float
    height: float


class AnnotationUpdate(BaseModel):
    boxes: list[Box] = Field(default_factory=list)


class ReviewUpdate(BaseModel):
    status: str


class CommitRequest(BaseModel):
    scope: str = "ready"
    indices: list[int] = Field(default_factory=list)
    operator: str = "human"


class DatasetRegisterRequest(BaseModel):
    id: str
    name: str
    path: str
    source_type: str = "self_collected"
    read_only: bool = True
    exhaustive_classes: list[int] = Field(default_factory=list)


class ScanRequest(BaseModel):
    dataset_ids: list[str] = Field(default_factory=list)


class BuildRequest(BaseModel):
    dataset_ids: list[str]
    selected_classes: list[int]
    targets: dict[int, int] = Field(default_factory=dict)
    dataset_targets: dict[str, dict[int, int]] = Field(default_factory=dict)
    output: str
    seed: int = 0
    source_ratios: dict[str, float] = Field(default_factory=dict)
    split_ratio: tuple[float, float, float] = (0.7, 0.2, 0.1)
    dry_run: bool = False


class AutoLabelRequest(BaseModel):
    replace_candidates: bool = True
    provider_id: str = "openai"


class BatchRequest(BaseModel):
    scope: str = "unreviewed"
    indices: list[int] = Field(default_factory=list)
    candidate_policy: str = "overwrite"  # skip_existing | overwrite
    provider_id: str = "openai"


class ReviewBatchRequest(BaseModel):
    action: str = "mark_ready"  # mark_ready | withdraw
    scope: str = "drafts"  # drafts | ready | selected | filtered
    indices: list[int] = Field(default_factory=list)
    query: str = ""
    status: str = ""


class DraftBatchRequest(BaseModel):
    source: str = "official"  # official | candidate | empty
    scope: str = "pending"  # pending | unreviewed | all | selected | filtered
    indices: list[int] = Field(default_factory=list)
    query: str = ""
    status: str = ""
    overwrite: bool = False
    exclude_reviewed: bool = False


def create_app(
    dataset_root: Path,
    class_config: Path,
    provider: AutoLabelProvider | None = None,
    registry_path: Path | None = None,
    rescan: bool = False,
    settings_dir: Path | None = None,
    task_id: str = "",
) -> FastAPI:
    classes = load_classes(class_config)
    dataset = YoloDataset(dataset_root, classes, rescan=rescan)
    registry_path = registry_path or (dataset_root.parent / "datasets_registry.yaml")
    cache_dir = registry_path.parent / ".d1_dataset_scan_cache"
    app = FastAPI(title="D1 Dataset Annotation", version="0.3.0")
    app.state.dataset = dataset
    app.state.settings_dir = settings_dir
    app.state.task_id = task_id or ""
    app.state.task_system_prompt = os.getenv("D1_TASK_SYSTEM_PROMPT", "")
    app.state.task_key_id = os.getenv("D1_TASK_KEY_ID", "")
    app.state.default_provider_id = os.getenv("D1_DEFAULT_PROVIDER_ID", "openai")
    conf_env = os.getenv("D1_TASK_CONFIDENCE")
    app.state.task_confidence = float(conf_env) if conf_env else None
    jobs = JobManager(max_workers=1)
    scan_jobs = ScanJobManager()
    build_jobs = BuildJobManager()
    review_jobs = ReviewJobManager()
    app.state.jobs = jobs
    app.state.scan_jobs = scan_jobs
    app.state.build_jobs = build_jobs
    app.state.review_jobs = review_jobs
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    def resolve_provider(provider_id: str) -> AutoLabelProvider:
        if provider is not None:
            return provider
        return create_provider(
            provider_id,
            settings_dir=app.state.settings_dir,
            system_prompt=app.state.task_system_prompt or None,
            key_id=app.state.task_key_id or None,
            confidence_threshold=app.state.task_confidence,
        )

    @app.middleware("http")
    async def no_cache_api(request: Request, call_next):
        response = await call_next(request)
        if (request.url.path.startswith("/api/")
                or request.url.path.startswith("/static/")
                or request.url.path in {"/", "/builder"}):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    @app.exception_handler(DatasetError)
    async def dataset_error_handler(_request: Any, exc: DatasetError):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/builder")
    def builder():
        return FileResponse(STATIC_DIR / "builder.html")

    def cached_scan(source: DatasetSource, refresh: bool = False):
        if not refresh:
            cached = load_scan_cache(cache_dir, source)
            if cached is not None:
                return cached, True
        scan = scan_source(source, classes)
        save_scan_cache(cache_dir, scan)
        return scan, False

    @app.post("/api/datasets/scan")
    def start_dataset_scan(request: ScanRequest = ScanRequest()):
        registered = load_registry(registry_path)
        by_id = {source.id: source for source in registered}
        requested_ids = request.dataset_ids or [source.id for source in registered]
        unknown = [source_id for source_id in requested_ids if source_id not in by_id]
        if unknown:
            raise HTTPException(status_code=422, detail=f"未知数据集：{', '.join(unknown)}")
        sources = [by_id[source_id] for source_id in requested_ids]
        if not sources:
            raise HTTPException(status_code=422, detail="没有登记数据集")
        active = scan_jobs.submit(sources, lambda source, _index, _total, progress: save_scan_cache(
            cache_dir, scan_source(source, classes, progress=progress)))
        return active

    @app.post("/api/datasets/scan/{job_id}/stop")
    def stop_dataset_scan(job_id: str):
        job = scan_jobs.cancel(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="扫描任务不存在")
        return job

    @app.get("/api/datasets/scan/{job_id}")
    def dataset_scan_status(job_id: str):
        job = scan_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="扫描任务不存在")
        return job

    @app.get("/api/datasets")
    def datasets(refresh: bool = False):
        sources = load_registry(registry_path)
        rows = []
        missing = []
        for source in sources:
            try:
                scan = scan_source(source, classes) if refresh else load_scan_cache(cache_dir, source)
                if scan is None:
                    missing.append(source.id)
                    rows.append({"id": source.id, "name": source.name, "path": str(source.root),
                                 "source_type": source.source_type, "cached": False,
                                 "cache_ready": False, "needs_rescan": True})
                    continue
                if refresh:
                    save_scan_cache(cache_dir, scan)
                row = {k: v for k, v in scan.items() if k != "samples"}
                row.update({"cached": not refresh, "cache_ready": True, "needs_rescan": False})
                rows.append(row)
            except DatasetError as exc:
                rows.append({"id": source.id, "name": source.name, "path": str(source.root), "error": str(exc)})
        return {"registry": str(registry_path), "datasets": rows,
                "ready": not missing, "needs_rescan": missing}

    @app.post("/api/datasets")
    def register_dataset(request: DatasetRegisterRequest):
        sources = load_registry(registry_path)
        if any(s.id == request.id for s in sources):
            raise HTTPException(status_code=409, detail="Dataset id already exists")
        source = DatasetSource(request.id, request.name, Path(request.path).expanduser().resolve(), request.source_type, request.read_only, tuple(request.exhaustive_classes))
        if not source.root.is_dir():
            raise HTTPException(status_code=422, detail="Dataset path does not exist")
        sources.append(source); save_registry(registry_path, sources)
        return {"ok": True, "id": source.id}

    @app.get("/api/datasets/summary")
    def dataset_summary(dataset_ids: str = Query(default="")):
        # Summary must be fast and read-only. Never silently scan a large dataset
        # from this button; the user should explicitly click “扫描数据集”.
        sources = load_registry(registry_path)
        by_id = {source.id: source for source in sources}
        selected_ids = [item.strip() for item in dataset_ids.split(",") if item.strip()]
        if not selected_ids:
            selected_ids = [source.id for source in sources]
        unknown = [source_id for source_id in selected_ids if source_id not in by_id]
        if unknown:
            raise HTTPException(status_code=422, detail=f"未知数据集：{', '.join(unknown)}")

        scans = []
        missing = []
        for source_id in selected_ids:
            source = by_id[source_id]
            scan = load_scan_cache(cache_dir, source)
            if scan is None:
                missing.append(source.id)
            else:
                scans.append(scan)
        if missing:
            return {"ready": False, "needs_rescan": missing,
                    "message": f"以下数据集没有当前版本的扫描缓存：{', '.join(missing)}。请先点击重新扫描。",
                    "datasets": [], "classes": {}, "quality": {}}
        result = aggregate_scans(scans, classes)
        result.update({"ready": True, "needs_rescan": [], "message": "扫描缓存已就绪"})
        return result

    @app.post("/api/datasets/build/start")
    def start_dataset_build(request: BuildRequest):
        sources = {source.id: source for source in load_registry(registry_path)}
        selected = [sources[item] for item in request.dataset_ids if item in sources]
        if len(selected) != len(request.dataset_ids):
            raise HTTPException(status_code=422, detail="Unknown dataset id")
        scans = []
        for source in selected:
            scan = load_scan_cache(cache_dir, source)
            if scan is None:
                raise HTTPException(status_code=422, detail=f"数据集 {source.id} 没有当前版本扫描缓存，请先重新扫描")
            scans.append(scan)
        targets = {int(k): int(v) for k, v in request.targets.items()}
        def worker(progress):
            return build_dataset(scans, classes, Path(request.output).expanduser().resolve(),
                                 request.selected_classes, targets, request.seed,
                                 split_ratio=request.split_ratio,
                                 source_ratios=request.source_ratios,
                                 dataset_targets=request.dataset_targets,
                                 dry_run=request.dry_run, progress=progress)
        return build_jobs.submit(worker, request.dry_run)

    @app.get("/api/datasets/build/{job_id}")
    def dataset_build_status(job_id: str):
        job = build_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="构建任务不存在")
        return job

    @app.post("/api/datasets/build")
    def dataset_build(request: BuildRequest):
        sources = {source.id: source for source in load_registry(registry_path)}
        selected = [sources[item] for item in request.dataset_ids if item in sources]
        if len(selected) != len(request.dataset_ids):
            raise HTTPException(status_code=422, detail="Unknown dataset id")
        scans = [cached_scan(source, False)[0] for source in selected]
        targets = {int(k): int(v) for k, v in request.targets.items()}
        try:
            report = build_dataset(
                scans, classes, Path(request.output).expanduser().resolve(),
                request.selected_classes, targets, request.seed,
                split_ratio=request.split_ratio,
                source_ratios=request.source_ratios,
                dataset_targets=request.dataset_targets,
                dry_run=request.dry_run,
            )
        except (DatasetError, OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return report

    @app.post("/api/images/rescan")
    def rescan_images():
        return {"ok": True, "count": dataset.rescan_images(), "cache": str(dataset.image_index_path)}

    @app.get("/api/config")
    def config():
        catalog = provider_catalog(provider, settings_dir=app.state.settings_dir)
        preferred = app.state.default_provider_id
        active = next((x for x in catalog if x.get("id") == preferred), None) or (catalog[0] if catalog else {})
        return {
            "classes": classes,
            "autolabel": active,
            "autolabel_providers": catalog,
            "default_provider_id": preferred,
            "task_id": app.state.task_id,
            **dataset.summary(),
        }

    @app.get("/api/autolabel/providers")
    def autolabel_providers():
        return {
            "providers": provider_catalog(provider, settings_dir=app.state.settings_dir),
            "default_provider_id": app.state.default_provider_id,
        }

    @app.get("/api/images")
    def images(page: int | None = None, page_size: int = 200, q: str = "", status: str = ""):
        # Keep the legacy unpaged response for API compatibility. The Web UI
        # always requests a page, avoiding a 100k+ JSON response and DOM nodes.
        if page is None:
            return dataset.list_images()
        return dataset.image_page(page=page, page_size=page_size, query=q, status=status)

    @app.get("/api/images/{index}/file")
    def image_file(index: int):
        try:
            return FileResponse(dataset.item(index).image)
        except IndexError as exc:
            raise HTTPException(status_code=404, detail="Image not found") from exc

    @app.get("/api/images/{index}/annotations")
    def annotations(index: int):
        try:
            return {"image": dataset.image_info(index), "boxes": dataset.read_boxes(index), "candidates": dataset.read_candidates(index), "history": dataset.history_versions(index), "draft": dataset.read_draft(index), "review_status": dataset.image_info(index)["status"]}
        except IndexError as exc:
            raise HTTPException(status_code=404, detail="Image not found") from exc

    @app.get("/api/images/{index}/history/{version_index}")
    def history_version(index: int, version_index: int):
        try:
            return {"boxes": dataset.read_history(index, version_index)}
        except IndexError as exc:
            raise HTTPException(status_code=404, detail="History version not found") from exc

    @app.put("/api/images/{index}/annotations")
    def update_annotations(index: int, update: AnnotationUpdate):
        try:
            dataset.save_boxes(index, [box.model_dump() for box in update.boxes])
            return {"ok": True, "boxes": len(update.boxes)}
        except IndexError as exc:
            raise HTTPException(status_code=404, detail="Image not found") from exc

    @app.post("/api/images/{index}/autolabel")
    def autolabel_image(index: int, request: AutoLabelRequest):
        try:
            active_provider = resolve_provider(request.provider_id)
            boxes = active_provider.predict(dataset.item(index).image, classes)
            dataset.save_candidates(index, boxes, active_provider.name, active_provider.model)
            return {"ok": True, "provider": active_provider.name, "model": active_provider.model, "boxes": boxes}
        except IndexError as exc:
            raise HTTPException(status_code=404, detail="Image not found") from exc
        except AutoLabelError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except Exception as exc:
            # Keep unexpected internals private while making known relay errors actionable.
            raise HTTPException(status_code=502, detail=f"自动标注发生未预期错误：{type(exc).__name__}") from exc

    @app.get("/api/autolabel/batch-counts")
    def autolabel_batch_counts():
        return {
            "all": len(dataset.batch_indices("all")),
            "unreviewed": len(dataset.batch_indices("unreviewed")),
            "unlabeled": len(dataset.batch_indices("unlabeled")),
        }

    @app.post("/api/autolabel/batch")
    def autolabel_batch(request: BatchRequest):
        try:
            active_provider = resolve_provider(request.provider_id)
        except AutoLabelError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if request.scope == "selected":
            indices = sorted(set(request.indices))
        else:
            try:
                indices = dataset.batch_indices(request.scope)
            except DatasetError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        if any(i < 0 or i >= len(dataset.items) for i in indices):
            raise HTTPException(status_code=422, detail="Batch contains invalid image index")
        if request.candidate_policy not in {"skip_existing", "overwrite"}:
            raise HTTPException(status_code=422, detail="candidate_policy must be skip_existing or overwrite")
        # Do not reject a valid selection merely because it exceeds the old
        # safety limit. The job manager already processes images one by one,
        # so large selections remain resumable and their progress is visible.
        # The former D1_AUTOLABEL_BATCH_LIMIT is intentionally no longer a
        # hard rejection: the background job processes one image at a time.

        def worker(index: int) -> int:
            if request.candidate_policy == "skip_existing" and dataset.read_candidates(index) is not None:
                return 0
            boxes = active_provider.predict(dataset.item(index).image, classes)
            dataset.save_candidates(index, boxes, active_provider.name, active_provider.model,
                                    overwrite=request.candidate_policy == "overwrite")
            return len(boxes)

        return jobs.submit(indices, worker)

    @app.post("/api/autolabel/jobs/{job_id}/stop")
    def stop_autolabel_job(job_id: str):
        job = jobs.cancel(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.get("/api/autolabel/jobs/{job_id}")
    def autolabel_job(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.delete("/api/images/{index}/candidates")
    def clear_candidates(index: int):
        try:
            dataset.item(index)
            dataset.clear_candidates(index)
            return {"ok": True}
        except IndexError as exc:
            raise HTTPException(status_code=404, detail="Image not found") from exc

    @app.put("/api/images/{index}/draft")
    def save_draft(index: int, update: AnnotationUpdate):
        try:
            for box in update.boxes:
                dataset._validate_box(box.class_id, box.x, box.y, box.width, box.height)
            dataset.save_draft(index, [box.model_dump() for box in update.boxes])
            dataset.set_review(index, "draft_saved")
            return {"ok": True, "status": "draft_saved", "boxes": len(update.boxes)}
        except IndexError as exc:
            raise HTTPException(status_code=404, detail="Image not found") from exc

    @app.get("/api/commit/preview")
    def commit_preview(scope: str = "ready", indices: str = ""):
        if scope == "ready":
            selected = [i for i, item in enumerate(dataset.items) if dataset._review.get(item.key) == "draft_ready"]
        elif scope == "selected":
            try:
                selected = [int(x) for x in indices.split(",") if x.strip()]
            except ValueError as exc:
                raise HTTPException(status_code=422, detail="indices must be comma-separated integers") from exc
            if any(i < 0 or i >= len(dataset.items) for i in selected):
                raise HTTPException(status_code=422, detail="Commit contains invalid image index")
            selected = [i for i in selected if dataset._review.get(dataset.item(i).key) == "draft_ready"]
        else:
            raise HTTPException(status_code=422, detail="scope must be ready or selected")
        return dataset.commit_preview(selected)

    @app.post("/api/commit")
    def commit_drafts(request: CommitRequest):
        if request.scope == "ready":
            selected = [i for i, item in enumerate(dataset.items) if dataset._review.get(item.key) == "draft_ready"]
        elif request.scope == "selected":
            selected = sorted(set(request.indices))
        else:
            raise HTTPException(status_code=422, detail="scope must be ready or selected")
        if any(i < 0 or i >= len(dataset.items) for i in selected):
            raise HTTPException(status_code=422, detail="Commit contains invalid image index")
        if request.scope == "selected":
            selected = [i for i in selected if dataset._review.get(dataset.item(i).key) == "draft_ready"]
        return dataset.commit_drafts(selected, request.operator[:80])

    def review_selection(scope: str, indices: list[int], query: str, status: str, action: str) -> tuple[list[int], dict[str, int]]:
        if action not in {"mark_ready", "withdraw"}:
            raise HTTPException(status_code=422, detail="action must be mark_ready or withdraw")
        if scope == "selected":
            selected = sorted(set(indices))
        elif scope == "filtered":
            normalized_query = query.strip().casefold()
            valid_statuses = {"", "pending", "draft_saved", "draft_ready", "reviewed", "needs_review"}
            if status not in valid_statuses:
                raise HTTPException(status_code=422, detail="Invalid review status filter")
            selected = [
                i for i, item in enumerate(dataset.items)
                if (not normalized_query or normalized_query in item.key.casefold())
                and (not status or dataset._review.get(item.key, "pending") == status)
            ]
        elif scope == "drafts":
            selected = [i for i, item in enumerate(dataset.items) if dataset.read_draft(i) is not None]
        elif scope == "ready":
            selected = [i for i, item in enumerate(dataset.items) if dataset._review.get(item.key) == "draft_ready"]
        else:
            raise HTTPException(status_code=422, detail="scope must be drafts, ready, filtered, or selected")
        if any(i < 0 or i >= len(dataset.items) for i in selected):
            raise HTTPException(status_code=422, detail="Review contains invalid image index")
        target = "draft_ready" if action == "mark_ready" else "draft_saved"
        eligible = []
        skipped = {"no_draft": 0, "needs_review": 0, "already_target": 0, "wrong_status": 0}
        for i in selected:
            current = dataset._review.get(dataset.item(i).key, "pending")
            if action == "mark_ready":
                if dataset.read_draft(i) is None:
                    skipped["no_draft"] += 1
                elif current == "needs_review":
                    skipped["needs_review"] += 1
                elif current == target:
                    skipped["already_target"] += 1
                elif current == "reviewed":
                    skipped["wrong_status"] += 1
                else:
                    eligible.append(i)
            else:
                if current == target:
                    skipped["already_target"] += 1
                elif current != "draft_ready":
                    skipped["wrong_status"] += 1
                else:
                    eligible.append(i)
        return eligible, skipped

    def draft_selection(scope: str, indices: list[int], query: str, status: str) -> list[int]:
        if scope == "selected":
            selected = sorted(set(indices))
        elif scope == "pending":
            selected = [i for i, item in enumerate(dataset.items)
                        if dataset._review.get(item.key, "pending") == "pending"]
        elif scope == "unreviewed":
            selected = [i for i, item in enumerate(dataset.items)
                        if dataset._review.get(item.key, "pending") != "reviewed"]
        elif scope == "all":
            selected = list(range(len(dataset.items)))
        elif scope == "filtered":
            normalized_query = query.strip().casefold()
            valid_statuses = {"", "pending", "draft_saved", "draft_ready", "reviewed", "needs_review"}
            if status not in valid_statuses:
                raise HTTPException(status_code=422, detail="Invalid review status filter")
            selected = [i for i, item in enumerate(dataset.items)
                        if (not normalized_query or normalized_query in item.key.casefold())
                        and (not status or dataset._review.get(item.key, "pending") == status)]
        else:
            raise HTTPException(status_code=422, detail="scope must be pending, unreviewed, all, filtered, or selected")
        if any(i < 0 or i >= len(dataset.items) for i in selected):
            raise HTTPException(status_code=422, detail="Draft contains invalid image index")
        return selected

    def draft_preview(source: str, selected: list[int], overwrite: bool,
                      exclude_reviewed: bool = False) -> tuple[list[int], dict[str, int], int]:
        if source not in {"official", "candidate", "empty"}:
            raise HTTPException(status_code=422, detail="source must be official, candidate, or empty")
        eligible = []
        skipped = {"existing_draft": 0, "no_candidate": 0, "reviewed": 0}
        overwriting = 0
        for index in selected:
            current = dataset._review.get(dataset.item(index).key, "pending")
            existing = dataset.read_draft(index) is not None
            if exclude_reviewed and current == "reviewed":
                skipped["reviewed"] += 1
            elif existing and not overwrite:
                skipped["existing_draft"] += 1
            elif source == "candidate" and dataset.read_candidates(index) is None:
                skipped["no_candidate"] += 1
            else:
                eligible.append(index)
                if existing:
                    overwriting += 1
        return eligible, skipped, overwriting

    @app.get("/api/drafts/batch/preview")
    def draft_batch_preview(source: str = "official", scope: str = "pending",
                            indices: str = "", q: str = "", status: str = "",
                            overwrite: bool = False, exclude_reviewed: bool = False):
        try:
            parsed = [int(x) for x in indices.split(",") if x.strip()]
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="indices must be comma-separated integers") from exc
        selected = draft_selection(scope, parsed, q, status)
        # Candidate replacement is explicitly allowed to reopen reviewed images;
        # keep preview consistent with the POST endpoint below.
        exclude_reviewed = exclude_reviewed and not (source == "candidate" and overwrite)
        eligible, skipped, overwriting = draft_preview(source, selected, overwrite, exclude_reviewed)
        return {"source": source, "scope": scope, "selected": len(selected),
                "eligible": len(eligible), "skipped": skipped,
                "overwriting_existing": overwriting, "indices": eligible}

    @app.post("/api/drafts/batch")
    def draft_batch(request: DraftBatchRequest):
        selected = draft_selection(request.scope, request.indices, request.query, request.status)
        # Re-running the model on an already submitted image must be allowed:
        # replacing its candidate with a new draft intentionally reopens review.
        exclude_reviewed = request.exclude_reviewed and not (
            request.source == "candidate" and request.overwrite
        )
        eligible, skipped, overwriting = draft_preview(
            request.source, selected, request.overwrite, exclude_reviewed)
        def worker(progress):
            result = dataset.create_drafts(eligible, request.source, request.overwrite, progress)
            result["skipped"] = skipped
            result["overwriting_existing"] = overwriting
            if request.source == "candidate" and request.overwrite:
                invalidated = [
                    index for index in eligible
                    if dataset._review.get(dataset.item(index).key, "pending") in {"draft_ready", "reviewed"}
                ]
                if invalidated:
                    dataset.set_reviews(invalidated, "draft_saved")
                result["invalidated_ready"] = len(invalidated)
            return result
        return review_jobs.submit(eligible, "create_draft", worker)

    @app.get("/api/review/batch/preview")
    def review_batch_preview(action: str = "mark_ready", scope: str = "drafts",
                             indices: str = "", q: str = "", status: str = ""):
        try:
            parsed = [int(x) for x in indices.split(",") if x.strip()]
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="indices must be comma-separated integers") from exc
        eligible, skipped = review_selection(scope, parsed, q, status, action)
        return {"action": action, "scope": scope, "selected": len(eligible) + sum(skipped.values()),
                "eligible": len(eligible), "skipped": skipped, "indices": eligible}

    @app.post("/api/review/batch")
    def review_batch(request: ReviewBatchRequest):
        eligible, skipped = review_selection(request.scope, request.indices, request.query, request.status, request.action)
        action_status = "draft_ready" if request.action == "mark_ready" else "draft_saved"
        def worker(progress):
            result = dataset.set_reviews(eligible, action_status, progress)
            result["skipped"] = skipped
            return result
        return review_jobs.submit(eligible, request.action, worker)

    @app.get("/api/review/jobs/{job_id}")
    def review_job(job_id: str):
        job = review_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Review job not found")
        return job

    @app.put("/api/images/{index}/review")
    def update_review(index: int, update: ReviewUpdate):
        try:
            dataset.set_review(index, update.status)
            return {"ok": True, "status": update.status}
        except IndexError as exc:
            raise HTTPException(status_code=404, detail="Image not found") from exc

    return app
