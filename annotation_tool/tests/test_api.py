from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from d1_annotation.app import create_app

CLASSES = Path(__file__).parents[1] / "config" / "classes.yaml"


def test_annotation_api(tmp_path):
    image = tmp_path / "images" / "val" / "frame.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (320, 240), "white").save(image)
    client = TestClient(create_app(tmp_path, CLASSES))
    config = client.get("/api/config").json()
    assert config["count"] == 1
    assert config["classes"][12]["name"] == "cigarette_or_smoke"
    payload = {"boxes": [{"class_id": 12, "x": .5, "y": .5, "width": .05, "height": .1}]}
    assert client.put("/api/images/0/annotations", json=payload).status_code == 200
    assert client.get("/api/images/0/annotations").json()["boxes"] == payload["boxes"]
    assert client.put("/api/images/0/review", json={"status": "reviewed"}).status_code == 200
    assert (tmp_path / "labels" / "val" / "frame.txt").exists()


def test_annotation_page_has_persistent_autolabel_progress(tmp_path):
    image = tmp_path / "images" / "train" / "frame.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (64, 64), "black").save(image)
    client = TestClient(create_app(tmp_path, CLASSES))
    page = client.get("/").text
    assert 'id="autolabelProgress"' in page
    assert 'id="autolabelBar"' in page
    assert 'id="autolabelProgressText"' in page
    assert "批量自动标注" in page

class FakeVisionProvider:
    name = "fake-vision"
    model = "fake-v1"

    def predict(self, image, classes):
        assert image.exists()
        assert classes[12]["name"] == "cigarette_or_smoke"
        return [{
            "class_id": 13, "x": .5, "y": .5, "width": .3, "height": .4,
            "confidence": .88, "evidence": "visible dog", "source": "fake_api",
        }]


def test_model_candidate_requires_human_acceptance(tmp_path):
    image = tmp_path / "images" / "train" / "dog.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (200, 100), "gray").save(image)
    client = TestClient(create_app(tmp_path, CLASSES, provider=FakeVisionProvider()))
    response = client.post("/api/images/0/autolabel", json={"replace_candidates": True})
    assert response.status_code == 200
    assert response.json()["boxes"][0]["class_id"] == 13
    # Model output is a candidate sidecar and must not become an official label.
    assert not (tmp_path / "labels" / "train" / "dog.txt").exists()
    annotations = client.get("/api/images/0/annotations").json()
    assert annotations["boxes"] == []
    assert annotations["candidates"]["model"] == "fake-v1"
    assert annotations["candidates"]["boxes"][0]["confidence"] == .88
    assert client.delete("/api/images/0/candidates").status_code == 200
    assert client.get("/api/images/0/annotations").json()["candidates"] is None


def test_provider_catalog_and_request_selection(tmp_path):
    image = tmp_path / "images" / "train" / "dog.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (200, 100), "gray").save(image)
    client = TestClient(create_app(tmp_path, CLASSES, provider=FakeVisionProvider()))
    config = client.get("/api/config").json()
    assert config["autolabel_providers"][0]["configured"] is True
    response = client.post("/api/images/0/autolabel", json={
        "replace_candidates": True,
        "provider_id": "sam3",
    })
    # An injected provider remains the deterministic provider in tests.
    assert response.status_code == 200
    assert response.json()["provider"] == "fake-vision"


def test_batch_autolabel_job(tmp_path):
    import time
    for name in ("a.jpg", "b.jpg"):
        image = tmp_path / "images" / "train" / name
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), "black").save(image)
    client = TestClient(create_app(tmp_path, CLASSES, provider=FakeVisionProvider()))
    job = client.post("/api/autolabel/batch", json={"scope": "unlabeled", "indices": []}).json()
    for _ in range(100):
        status = client.get(f"/api/autolabel/jobs/{job['id']}").json()
        if status["status"].startswith("completed"):
            break
        time.sleep(.01)
    assert status["status"] == "completed"
    assert status["completed"] == 2
    assert status["boxes"] == 2
    assert len(list((tmp_path / ".d1_annotations" / "candidates").rglob("*.json"))) == 2


def test_batch_autolabel_can_be_stopped(tmp_path):
    import time

    class SlowProvider(FakeVisionProvider):
        def predict(self, image, classes):
            time.sleep(0.05)
            return super().predict(image, classes)

    for name in ("a.jpg", "b.jpg", "c.jpg"):
        image = tmp_path / "images" / "train" / name
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), "black").save(image)
    client = TestClient(create_app(tmp_path, CLASSES, provider=SlowProvider()))
    started = client.post("/api/autolabel/batch", json={"scope": "all", "candidate_policy": "overwrite"})
    assert started.status_code == 200
    stopped = client.post(f"/api/autolabel/jobs/{started.json()['id']}/stop")
    assert stopped.status_code == 200
    for _ in range(100):
        job = client.get(f"/api/autolabel/jobs/{started.json()['id']}").json()
        if job["status"] in {"cancelled", "completed", "completed_with_errors"}: break
        time.sleep(.01)
    assert job["status"] == "cancelled"
    assert job["completed"] < job["total"]


def test_batch_all_includes_already_labeled_images(tmp_path, monkeypatch):
    import time
    for name in ("old.jpg", "new.jpg"):
        image = tmp_path / "images" / "train" / name
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), "black").save(image)
    label = tmp_path / "labels" / "train" / "old.txt"
    label.parent.mkdir(parents=True)
    label.write_text("0 0.5 0.5 0.2 0.2\n")
    monkeypatch.setenv("D1_AUTOLABEL_BATCH_LIMIT", "10")
    client = TestClient(create_app(tmp_path, CLASSES, provider=FakeVisionProvider()))
    job = client.post("/api/autolabel/batch", json={"scope": "all", "indices": []}).json()
    for _ in range(100):
        status = client.get(f"/api/autolabel/jobs/{job['id']}").json()
        if status["status"].startswith("completed"):
            break
        time.sleep(.01)
    assert status["completed"] == 2
    assert label.read_text() == "0 0.5 0.5 0.2 0.2\n"
    assert len(list((tmp_path / ".d1_annotations" / "candidates").rglob("*.json"))) == 2


def test_history_api_returns_read_only_old_label(tmp_path):
    image = tmp_path / "images" / "train" / "frame.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (100, 100), "white").save(image)
    label = tmp_path / "labels" / "train" / "frame.txt"
    label.parent.mkdir(parents=True)
    label.write_text("7 0.500000 0.500000 0.200000 0.200000\n")
    client = TestClient(create_app(tmp_path, CLASSES))
    replacement = {"boxes": [{"class_id": 8, "x": .4, "y": .4, "width": .1, "height": .1}]}
    assert client.put("/api/images/0/annotations", json=replacement).status_code == 200
    annotation = client.get("/api/images/0/annotations").json()
    assert len(annotation["history"]) == 1
    old = client.get("/api/images/0/history/0").json()["boxes"]
    assert old == [{"class_id": 7, "x": .5, "y": .5, "width": .2, "height": .2}]
    assert client.get("/api/images/0/annotations").json()["boxes"] == replacement["boxes"]


def test_batch_counts_and_unreviewed_scope(tmp_path):
    for name in ("reviewed.jpg", "pending.jpg"):
        image = tmp_path / "images" / "train" / name
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), "black").save(image)
    client = TestClient(create_app(tmp_path, CLASSES, provider=FakeVisionProvider()))
    client.put("/api/images/0/review", json={"status": "reviewed"})
    counts = client.get("/api/autolabel/batch-counts").json()
    assert counts == {"all": 2, "unreviewed": 1, "unlabeled": 2}


def test_batch_policy_validation(tmp_path):
    image = tmp_path / "images" / "train" / "a.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (64, 64), "black").save(image)
    client = TestClient(create_app(tmp_path, CLASSES, provider=FakeVisionProvider()))
    response = client.post("/api/autolabel/batch", json={"scope": "all", "candidate_policy": "bad"})
    assert response.status_code == 422
    assert "candidate_policy" in response.json()["detail"]


def test_batch_review_preview_and_job(tmp_path):
    import time
    for name in ("a.jpg", "b.jpg"):
        image = tmp_path / "images" / "train" / name
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), "black").save(image)
    client = TestClient(create_app(tmp_path, CLASSES))
    draft = {"boxes": [{"class_id": 7, "x": .4, "y": .4, "width": .1, "height": .1}]}
    assert client.put("/api/images/0/draft", json=draft).status_code == 200
    preview = client.get("/api/review/batch/preview?scope=drafts&action=mark_ready").json()
    assert preview["eligible"] == 1
    assert sum(preview["skipped"].values()) == 0
    started = client.post("/api/review/batch", json={"scope": "drafts", "action": "mark_ready"})
    assert started.status_code == 200
    for _ in range(100):
        job = client.get(f"/api/review/jobs/{started.json()['id']}").json()
        if job["status"] in {"completed", "failed"}:
            break
        time.sleep(.01)
    assert job["status"] == "completed"
    assert job["result"]["updated"] == 1
    assert client.get("/api/images/0/annotations").json()["review_status"] == "draft_ready"


def test_batch_create_drafts_from_official_labels(tmp_path):
    import time
    for name in ("labeled.jpg", "empty.jpg"):
        image = tmp_path / "images" / "train" / name
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), "black").save(image)
    label = tmp_path / "labels" / "train" / "labeled.txt"
    label.parent.mkdir(parents=True)
    label.write_text("7 0.5 0.5 0.2 0.2\n")
    client = TestClient(create_app(tmp_path, CLASSES))
    preview = client.get("/api/drafts/batch/preview?scope=pending&source=official").json()
    assert preview["eligible"] == 2
    started = client.post("/api/drafts/batch", json={"scope": "pending", "source": "official"})
    assert started.status_code == 200
    for _ in range(100):
        job = client.get(f"/api/review/jobs/{started.json()['id']}").json()
        if job["status"] in {"completed", "failed"}: break
        time.sleep(.01)
    assert job["status"] == "completed"
    assert job["result"]["created"] == 2
    images = client.get("/api/images").json()
    labeled_index = next(item["index"] for item in images if item["name"] == "labeled.jpg")
    annotations = client.get(f"/api/images/{labeled_index}/annotations").json()
    assert annotations["draft"]["boxes"][0]["class_id"] == 7
    assert annotations["review_status"] == "draft_saved"


def test_batch_create_drafts_from_candidate_skips_missing_candidate(tmp_path):
    import time
    for name in ("candidate.jpg", "missing.jpg"):
        image = tmp_path / "images" / "train" / name
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), "black").save(image)
    client = TestClient(create_app(tmp_path, CLASSES, provider=FakeVisionProvider()))
    assert client.post("/api/images/0/autolabel", json={"replace_candidates": True}).status_code == 200
    preview = client.get("/api/drafts/batch/preview?scope=pending&source=candidate").json()
    assert preview["eligible"] == 1
    assert preview["skipped"]["no_candidate"] == 1
    started = client.post("/api/drafts/batch", json={"scope": "pending", "source": "candidate"})
    for _ in range(100):
        job = client.get(f"/api/review/jobs/{started.json()['id']}").json()
        if job["status"] in {"completed", "failed"}: break
        time.sleep(.01)
    assert job["result"]["created"] == 1


def test_batch_candidate_to_new_draft_reopens_reviewed_images(tmp_path):
    import time
    for name in ("draft_ready.jpg", "reviewed.jpg"):
        image = tmp_path / "images" / "train" / name
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), "black").save(image)
    client = TestClient(create_app(tmp_path, CLASSES, provider=FakeVisionProvider()))
    old_draft = {"boxes": [{"class_id": 7, "x": .4, "y": .4, "width": .1, "height": .1}]}
    for index in (0, 1):
        assert client.post(f"/api/images/{index}/autolabel", json={"replace_candidates": True}).status_code == 200
        assert client.put(f"/api/images/{index}/draft", json=old_draft).status_code == 200
    assert client.put("/api/images/0/review", json={"status": "draft_ready"}).status_code == 200
    assert client.put("/api/images/1/review", json={"status": "reviewed"}).status_code == 200

    preview = client.get(
        "/api/drafts/batch/preview?scope=all&source=candidate&overwrite=true&exclude_reviewed=true"
    ).json()
    assert preview["eligible"] == 2
    assert preview["overwriting_existing"] == 2
    assert preview["skipped"]["reviewed"] == 0

    started = client.post("/api/drafts/batch", json={
        "scope": "all", "source": "candidate", "overwrite": True, "exclude_reviewed": True,
    })
    assert started.status_code == 200
    for _ in range(100):
        job = client.get(f"/api/review/jobs/{started.json()['id']}").json()
        if job["status"] in {"completed", "failed"}:
            break
        time.sleep(.01)
    assert job["status"] == "completed"
    assert job["result"]["created"] == 2
    assert job["result"]["overwritten"] == 2

    replaced = client.get("/api/images/0/annotations").json()
    assert replaced["draft"]["boxes"] == [{
        "class_id": 13, "x": .5, "y": .5, "width": .3, "height": .4,
    }]
    assert replaced["review_status"] == "draft_saved"
    reopened = client.get("/api/images/1/annotations").json()
    assert reopened["draft"]["boxes"] == replaced["draft"]["boxes"]
    assert reopened["review_status"] == "draft_saved"
    assert len(list((tmp_path / ".d1_annotations" / "draft_history").rglob("*.json"))) == 2


def test_batch_review_withdraw_only_ready(tmp_path):
    image = tmp_path / "images" / "train" / "a.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (64, 64), "black").save(image)
    client = TestClient(create_app(tmp_path, CLASSES))
    client.put("/api/images/0/review", json={"status": "draft_ready"})
    response = client.post("/api/review/batch", json={"scope": "ready", "action": "withdraw"})
    assert response.status_code == 200
    import time
    for _ in range(100):
        job = client.get(f"/api/review/jobs/{response.json()['id']}").json()
        if job["status"] in {"completed", "failed"}: break
        time.sleep(.01)
    assert job["result"]["updated"] == 1
    assert client.get("/api/images/0/annotations").json()["review_status"] == "draft_saved"


def test_draft_save_does_not_change_official_label_and_commit_does(tmp_path):
    image = tmp_path / "images" / "train" / "a.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (64, 64), "black").save(image)
    label = tmp_path / "labels" / "train" / "a.txt"
    label.parent.mkdir(parents=True)
    label.write_text("0 0.5 0.5 0.2 0.2\n")
    client = TestClient(create_app(tmp_path, CLASSES))
    draft = {"boxes": [{"class_id": 7, "x": .4, "y": .4, "width": .1, "height": .1}]}
    assert client.put("/api/images/0/draft", json=draft).status_code == 200
    assert label.read_text() == "0 0.5 0.5 0.2 0.2\n"
    assert client.put("/api/images/0/review", json={"status": "draft_ready"}).status_code == 200
    preview = client.get("/api/commit/preview?scope=ready").json()
    assert preview["count"] == 1
    result = client.post("/api/commit", json={"scope": "ready"}).json()
    assert result["committed"] == 1
    assert label.read_text().startswith("7 ")


def test_selected_commit_only_commits_ready_image(tmp_path):
    for name in ("ready.jpg", "saved.jpg"):
        image = tmp_path / "images" / "train" / name
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), "black").save(image)
    client = TestClient(create_app(tmp_path, CLASSES))
    draft = {"boxes": [{"class_id": 7, "x": .4, "y": .4, "width": .1, "height": .1}]}
    for index in (0, 1):
        assert client.put(f"/api/images/{index}/draft", json=draft).status_code == 200
    assert client.put("/api/images/0/review", json={"status": "draft_ready"}).status_code == 200

    ready_preview = client.get("/api/commit/preview?scope=selected&indices=0").json()
    saved_preview = client.get("/api/commit/preview?scope=selected&indices=1").json()
    assert ready_preview["count"] == 1
    assert saved_preview["count"] == 0

    skipped = client.post("/api/commit", json={"scope": "selected", "indices": [1]}).json()
    assert skipped["committed"] == 0
    committed = client.post("/api/commit", json={"scope": "selected", "indices": [0]}).json()
    assert committed["committed"] == 1
    assert client.get("/api/images/0/annotations").json()["review_status"] == "reviewed"
    assert client.get("/api/images/1/annotations").json()["review_status"] == "draft_saved"


def test_saved_draft_is_returned_on_reopen(tmp_path):
    image = tmp_path / "images" / "train" / "a.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (64, 64), "black").save(image)
    client = TestClient(create_app(tmp_path, CLASSES))
    draft = {"boxes": [{"class_id": 7, "x": .4, "y": .4, "width": .1, "height": .1}]}
    assert client.put("/api/images/0/draft", json=draft).status_code == 200
    data = client.get("/api/images/0/annotations").json()
    assert data["draft"]["boxes"] == draft["boxes"]
    assert data["review_status"] == "draft_saved"


def test_api_responses_are_no_cache(tmp_path):
    image = tmp_path / "images" / "train" / "a.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (64, 64), "black").save(image)
    client = TestClient(create_app(tmp_path, CLASSES))
    response = client.get("/api/images")
    assert response.headers["cache-control"] == "no-store, no-cache, must-revalidate, max-age=0"


def test_dataset_summary_requires_current_scan_cache(tmp_path):
    image = tmp_path / "images" / "train" / "frame.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32), "white").save(image)
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "version: 1\ndatasets:\n"
        f"  - id: source\n    name: source\n    path: {tmp_path}\n",
        encoding="utf-8",
    )
    client = TestClient(create_app(tmp_path, CLASSES, registry_path=registry))
    response = client.get("/api/datasets/summary")
    assert response.status_code == 200
    assert response.json()["ready"] is False
    assert response.json()["needs_rescan"] == ["source"]


def test_dataset_summary_reads_scan_cache(tmp_path):
    from d1_annotation.datasets import DatasetSource, save_scan_cache, scan_source
    image = tmp_path / "images" / "train" / "frame.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32), "white").save(image)
    label = tmp_path / "labels" / "train" / "frame.txt"
    label.parent.mkdir(parents=True)
    label.write_text("7 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "version: 1\ndatasets:\n"
        f"  - id: source\n    name: source\n    path: {tmp_path}\n",
        encoding="utf-8",
    )
    source = DatasetSource("source", "source", tmp_path)
    cache_dir = registry.parent / ".d1_dataset_scan_cache"
    save_scan_cache(cache_dir, scan_source(source, __import__("d1_annotation.core", fromlist=["load_classes"]).load_classes(CLASSES)))
    client = TestClient(create_app(tmp_path, CLASSES, registry_path=registry))
    response = client.get("/api/datasets/summary")
    assert response.status_code == 200
    assert response.json()["quality"]["valid_boxes"] == 1


def test_dataset_summary_filters_selected_datasets(tmp_path):
    from d1_annotation.core import load_classes
    from d1_annotation.datasets import DatasetSource, save_scan_cache, scan_source

    roots = [tmp_path / "first", tmp_path / "second"]
    for index, root in enumerate(roots, 1):
        image = root / "images" / "train" / "frame.jpg"
        image.parent.mkdir(parents=True)
        Image.new("RGB", (32, 32), "white").save(image)
        label = root / "labels" / "train" / "frame.txt"
        label.parent.mkdir(parents=True)
        label.write_text(("7 0.5 0.5 0.2 0.2\n" * index), encoding="utf-8")

    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "version: 1\ndatasets:\n"
        f"  - id: first\n    path: {roots[0]}\n"
        f"  - id: second\n    path: {roots[1]}\n",
        encoding="utf-8",
    )
    class_config = load_classes(CLASSES)
    cache_dir = registry.parent / ".d1_dataset_scan_cache"
    for source_id, root in zip(("first", "second"), roots):
        source = DatasetSource(source_id, source_id, root)
        save_scan_cache(cache_dir, scan_source(source, class_config))

    client = TestClient(create_app(roots[0], CLASSES, registry_path=registry))
    response = client.get("/api/datasets/summary?dataset_ids=second")
    assert response.status_code == 200
    data = response.json()
    assert [dataset["id"] for dataset in data["datasets"]] == ["second"]
    assert data["quality"]["valid_boxes"] == 2


def test_dataset_summary_only_requires_selected_cache(tmp_path):
    from d1_annotation.core import load_classes
    from d1_annotation.datasets import DatasetSource, save_scan_cache, scan_source

    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        image = root / "images" / "train" / "frame.jpg"
        image.parent.mkdir(parents=True)
        Image.new("RGB", (16, 16), "white").save(image)
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "version: 1\ndatasets:\n"
        f"  - id: first\n    path: {first}\n"
        f"  - id: second\n    path: {second}\n",
        encoding="utf-8",
    )
    source = DatasetSource("second", "second", second)
    save_scan_cache(registry.parent / ".d1_dataset_scan_cache", scan_source(source, load_classes(CLASSES)))

    client = TestClient(create_app(second, CLASSES, registry_path=registry))
    response = client.get("/api/datasets/summary?dataset_ids=second")
    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_builder_page_and_per_dataset_build_api(tmp_path):
    image = tmp_path / "images" / "train" / "frame.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32), "white").save(image)
    label = tmp_path / "labels" / "train" / "frame.txt"
    label.parent.mkdir(parents=True)
    label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    registry = tmp_path / "registry.yaml"
    registry.write_text("version: 1\ndatasets:\n" f"  - id: source\n    name: source\n    path: {tmp_path}\n", encoding="utf-8")
    client = TestClient(create_app(tmp_path, CLASSES, registry_path=registry))
    assert client.get("/builder").status_code == 200
    scan = client.get("/api/datasets?refresh=true")
    assert scan.status_code == 200
    report = client.post("/api/datasets/build", json={
        "dataset_ids": ["source"], "selected_classes": [0],
        "dataset_targets": {"source": {"0": 1}}, "targets": {},
        "output": str(tmp_path / "output"), "split_ratio": [0.7, 0.2, 0.1],
        "dry_run": True,
    })
    assert report.status_code == 200
    assert report.json()["selected_images"] == 1


def test_background_dataset_scan_reports_progress(tmp_path):
    image = tmp_path / "images" / "train" / "frame.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32), "white").save(image)
    registry = tmp_path / "registry.yaml"
    registry.write_text("version: 1\ndatasets:\n" f"  - id: source\n    name: source\n    path: {tmp_path}\n", encoding="utf-8")
    client = TestClient(create_app(tmp_path, CLASSES, registry_path=registry))
    started = client.post("/api/datasets/scan")
    assert started.status_code == 200
    job_id = started.json()["id"]
    import time
    for _ in range(100):
        status = client.get(f"/api/datasets/scan/{job_id}").json()
        if status["status"] == "completed":
            break
        time.sleep(.01)
    assert status["status"] == "completed"
    assert status["current"] == 1
    assert status["total"] == 1
    assert status["percent"] == 100


def test_background_dataset_build_reports_progress(tmp_path):
    import time
    image = tmp_path / "images" / "train" / "frame.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32), "white").save(image)
    label = tmp_path / "labels" / "train" / "frame.txt"
    label.parent.mkdir(parents=True)
    label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    registry = tmp_path / "registry.yaml"
    registry.write_text("version: 1\ndatasets:\n" f"  - id: source\n    name: source\n    path: {tmp_path}\n", encoding="utf-8")
    client = TestClient(create_app(tmp_path, CLASSES, registry_path=registry))
    scan_id = client.post("/api/datasets/scan").json()["id"]
    for _ in range(100):
        if client.get(f"/api/datasets/scan/{scan_id}").json()["status"] == "completed": break
        time.sleep(.01)
    started = client.post("/api/datasets/build/start", json={
        "dataset_ids": ["source"], "selected_classes": [0],
        "dataset_targets": {"source": {"0": 1}}, "targets": {},
        "output": str(tmp_path / "out"), "dry_run": True,
    })
    assert started.status_code == 200
    job_id = started.json()["id"]
    for _ in range(100):
        job = client.get(f"/api/datasets/build/{job_id}").json()
        if job["status"] in ("completed", "failed"): break
        time.sleep(.01)
    assert job["status"] == "completed"
    assert job["percent"] == 100
    assert job["result"]["selected_images"] == 1


def test_selective_scan_only_writes_requested_dataset_cache(tmp_path):
    import time
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        image = root / "images" / "train" / "frame.jpg"
        image.parent.mkdir(parents=True)
        Image.new("RGB", (16, 16), "white").save(image)
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "version: 1\ndatasets:\n"
        f"  - id: first\n    path: {first}\n"
        f"  - id: second\n    path: {second}\n",
        encoding="utf-8",
    )
    client = TestClient(create_app(second, CLASSES, registry_path=registry))
    started = client.post("/api/datasets/scan", json={"dataset_ids": ["second"]})
    assert started.status_code == 200
    job_id = started.json()["id"]
    for _ in range(100):
        job = client.get(f"/api/datasets/scan/{job_id}").json()
        if job["status"] in {"completed", "cancelled"}:
            break
        time.sleep(.01)
    cache = tmp_path / ".d1_dataset_scan_cache"
    assert (cache / "second.scan_cache.json").exists()
    assert not (cache / "first.scan_cache.json").exists()
