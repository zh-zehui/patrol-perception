import json
from pathlib import Path

import pytest
from PIL import Image

from d1_annotation.core import DatasetError, YoloDataset, load_classes

CLASSES = Path(__file__).parents[1] / "config" / "classes.yaml"


def dataset(tmp_path: Path) -> YoloDataset:
    image = tmp_path / "images" / "train" / "sample.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (100, 80), "black").save(image)
    return YoloDataset(tmp_path, load_classes(CLASSES))


def test_frozen_class_order():
    assert [c["name"] for c in load_classes(CLASSES)] == [
        "person", "bicycle", "car", "truck", "smoke", "flame", "ebike",
        "litter_bottle", "litter_plastic_bag", "litter_paper",
        "litter_cardboard", "fishing_rod", "cigarette_or_smoke", "pet",
    ]


def test_round_trip_and_history(tmp_path):
    ds = dataset(tmp_path)
    first = [{"class_id": 13, "x": .5, "y": .5, "width": .2, "height": .25}]
    ds.save_boxes(0, first)
    assert ds.read_boxes(0) == first
    ds.save_boxes(0, [{"class_id": 12, "x": .4, "y": .4, "width": .1, "height": .1}])
    history = list((tmp_path / ".d1_annotations" / "history").rglob("*.txt"))
    assert len(history) == 1
    assert history[0].read_text().startswith("13 ")


def test_rejects_out_of_bounds_box(tmp_path):
    ds = dataset(tmp_path)
    with pytest.raises(DatasetError, match="outside image"):
        ds.save_boxes(0, [{"class_id": 0, "x": .95, "y": .5, "width": .2, "height": .2}])


def test_review_state(tmp_path):
    ds = dataset(tmp_path)
    ds.set_review(0, "reviewed")
    assert ds.image_info(0)["status"] == "reviewed"
    assert ds.summary()["reviewed"] == 1


def test_candidate_sidecar_does_not_modify_yolo_label(tmp_path):
    ds = dataset(tmp_path)
    candidates = [{"class_id": 13, "x": .5, "y": .5, "width": .2, "height": .2,
                   "confidence": .7, "evidence": "cat", "source": "api"}]
    ds.save_candidates(0, candidates, "openai", "vision-test")
    assert ds.read_boxes(0) == []
    stored = ds.read_candidates(0)
    assert stored["provider"] == "openai"
    assert stored["boxes"] == candidates


def test_api_upload_is_resized_and_oriented(tmp_path):
    import base64
    from io import BytesIO
    from d1_annotation.autolabel import prepare_image_data_url
    image = tmp_path / "large.png"
    Image.new("RGB", (3000, 1000), "red").save(image)
    url = prepare_image_data_url(image, max_edge=512)
    assert url.startswith("data:image/jpeg;base64,")
    with Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1]))) as prepared:
        assert prepared.size == (512, 171)



def test_relay_provider_configuration_and_request_flags(tmp_path, monkeypatch):
    from d1_annotation.autolabel import OpenAIVisionProvider, provider_status

    monkeypatch.setenv("OPENAI_PROVIDER_NAME", "campus_sub2api")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://relay.example/v1")
    monkeypatch.setenv("OPENAI_VISION_MODEL", "codex.gpt-test")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "low")
    monkeypatch.setenv("OPENAI_STORE", "false")
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-real-key")
    provider = OpenAIVisionProvider()
    captured = []
    provider._request = lambda payload, endpoint: captured.append(payload) or {
        "output": [{"content": [{"type": "output_text", "text":
            '{"detections":[{"class_id":13,"x_min":100,"y_min":200,'
            '"x_max":500,"y_max":800,"confidence":0.8,"evidence":"dog"}]}'
        }]}]
    }
    image = tmp_path / "sample.jpg"
    Image.new("RGB", (32, 32), "white").save(image)
    boxes = provider.predict(image, load_classes(CLASSES))
    assert provider.base_url == "http://relay.example/v1"
    assert captured[0]["model"] == "codex.gpt-test"
    assert captured[0]["reasoning"] == {"effort": "low"}
    assert captured[0]["store"] is False
    assert captured[0]["text"]["format"]["type"] == "json_schema"
    assert captured[0]["input"][0]["content"][1]["type"] == "input_image"
    assert boxes[0]["class_id"] == 13
    assert boxes[0]["width"] == .4
    status = provider_status()
    assert status["provider"] == "campus_sub2api"
    assert status["wire_api"] == "responses"
    assert "api_key" not in status


def test_sam3_service_provider_converts_pixel_boxes(tmp_path, monkeypatch):
    from d1_annotation.autolabel import SAM3ServiceProvider, provider_catalog

    monkeypatch.setenv("SAM3_SERVICE_URL", "http://127.0.0.1:8783")
    provider = SAM3ServiceProvider()
    provider._request = lambda image, classes: {
        "model": "sam3",
        "width": 200,
        "height": 100,
        "detections": [{
            "class_id": 13,
            "x_min": 20,
            "y_min": 10,
            "x_max": 120,
            "y_max": 60,
            "confidence": 0.91,
            "prompt": "dog",
        }],
    }
    image = tmp_path / "dog.jpg"
    Image.new("RGB", (200, 100), "gray").save(image)
    boxes = provider.predict(image, load_classes(CLASSES))
    assert boxes == [{
        "class_id": 13,
        "x": .35,
        "y": .35,
        "width": .5,
        "height": .5,
        "confidence": .91,
        "evidence": "dog",
        "source": "sam3-local",
    }]
    catalog = provider_catalog(settings_dir=tmp_path / "settings")
    assert [item["id"] for item in catalog] == ["openai", "sam3"]
    assert catalog[1]["configured"] is True


def test_relay_envelope_and_validation_errors_are_actionable(tmp_path, monkeypatch):
    from d1_annotation.autolabel import AutoLabelError, OpenAIVisionProvider

    provider = OpenAIVisionProvider(api_key="test", base_url="http://relay.example/v1")
    image = tmp_path / "sample.jpg"
    Image.new("RGB", (32, 32), "white").save(image)
    provider._request = lambda payload, endpoint: {"output_text": "not json"}
    with pytest.raises(AutoLabelError, match="没有返回有效 JSON"):
        provider.predict(image, load_classes(CLASSES))
    provider._request = lambda payload, endpoint: {"output_text": '{"detections":[{"class_id":99}]}' }
    with pytest.raises(AutoLabelError, match="class_id"):
        provider.predict(image, load_classes(CLASSES))


def test_model_root_aliases_are_normalized():
    from d1_annotation.autolabel import _json_object
    item = [{"class_id": 13, "x_min": 1, "y_min": 2, "x_max": 3, "y_max": 4}]
    assert _json_object('{"objects": []}') == {"detections": []}
    assert _json_object('{"boxes": []}') == {"detections": []}
    assert _json_object('{"annotations": []}') == {"detections": []}
    assert _json_object('[{"class_id": 13}]') == {"detections": [{"class_id": 13}]}
    assert _json_object('{"detections": []}') == {"detections": []}
    assert _json_object('"{\\"detections\\": []}"') == {"detections": []}
    with pytest.raises(Exception, match="实际顶层字段"):
        _json_object('{"description": "no objects"}')


def test_detection_coordinate_aliases_are_normalized(tmp_path, monkeypatch):
    from d1_annotation.autolabel import OpenAIVisionProvider
    provider = OpenAIVisionProvider(api_key="test", base_url="http://relay.example/v1")
    image = tmp_path / "sample.jpg"
    Image.new("RGB", (32, 32), "white").save(image)
    provider._request = lambda payload, endpoint: {"output_text": '{"detections": ['
        '{"class_name":"bottle","bbox":[100,200,500,800],"score":0.7}'
        ']}' }
    boxes = provider.predict(image, load_classes(CLASSES))
    assert boxes[0]["class_id"] == 7
    assert boxes[0]["x"] == .3
    assert boxes[0]["y"] == .5
    assert boxes[0]["width"] == .4
    assert boxes[0]["height"] == .6


def test_center_size_coordinates_are_normalized(tmp_path):
    from d1_annotation.autolabel import OpenAIVisionProvider
    provider = OpenAIVisionProvider(api_key="test", base_url="http://relay.example/v1")
    image = tmp_path / "sample.jpg"
    Image.new("RGB", (32, 32), "white").save(image)
    provider._request = lambda payload, endpoint: {"output_text": '{"detections": ['
        '{"class":"pet","x":500,"y":500,"width":200,"height":400}'
        ']}' }
    boxes = provider.predict(image, load_classes(CLASSES))
    assert boxes[0]["class_id"] == 13
    assert boxes[0]["width"] == .2
    assert boxes[0]["height"] == .4


def test_normalized_xyxy_and_unknown_coordinate_message(tmp_path):
    from d1_annotation.autolabel import AutoLabelError, OpenAIVisionProvider
    provider = OpenAIVisionProvider(api_key="test", base_url="http://relay.example/v1")
    image = tmp_path / "sample.jpg"
    Image.new("RGB", (32, 32), "white").save(image)
    provider._request = lambda payload, endpoint: {"output_text": '{"detections": ['
        '{"label":"pet","xmin":0.1,"ymin":0.2,"xmax":0.5,"ymax":0.8}'
        ']}' }
    box = provider.predict(image, load_classes(CLASSES))[0]
    assert box["width"] == .4
    assert box["height"] == .6
    provider._request = lambda payload, endpoint: {"output_text": '{"detections": [{"label":"pet","polygon":[]}]}' }
    with pytest.raises(AutoLabelError, match="实际字段为：label, polygon"):
        provider.predict(image, load_classes(CLASSES))


def test_history_versions_can_be_listed_and_loaded(tmp_path):
    ds = dataset(tmp_path)
    original = [{"class_id": 7, "x": .5, "y": .5, "width": .2, "height": .2}]
    updated = [{"class_id": 8, "x": .4, "y": .4, "width": .1, "height": .1}]
    ds.save_boxes(0, original)
    ds.save_boxes(0, updated)
    versions = ds.history_versions(0)
    assert len(versions) == 1
    assert versions[0]["boxes"] == 1
    assert ds.read_history(0, versions[0]["index"]) == original
    assert ds.read_boxes(0) == updated


def test_batch_indices_follow_review_status(tmp_path):
    dataset(tmp_path)
    image2 = tmp_path / "images" / "train" / "second.jpg"
    Image.new("RGB", (100, 80), "black").save(image2)
    ds = YoloDataset(tmp_path, load_classes(CLASSES))
    assert ds.batch_indices("all") == [0, 1]
    assert ds.batch_indices("unreviewed") == [0, 1]
    ds.set_review(0, "reviewed")
    assert ds.batch_indices("unreviewed") == [1]
    assert ds.batch_indices("unlabeled") == [0, 1]


def test_candidate_overwrite_keeps_candidate_history(tmp_path):
    ds = dataset(tmp_path)
    first = [{"class_id": 7, "x": .5, "y": .5, "width": .2, "height": .2}]
    second = [{"class_id": 8, "x": .4, "y": .4, "width": .1, "height": .1}]
    assert ds.save_candidates(0, first, "fake", "v1", overwrite=True) is True
    assert ds.save_candidates(0, second, "fake", "v2", overwrite=True) is True
    history = list((tmp_path / ".d1_annotations" / "candidate_history").rglob("*.json"))
    assert len(history) == 1
    assert json.loads(history[0].read_text())["boxes"] == first
    assert ds.read_candidates(0)["boxes"] == second
    assert ds.save_candidates(0, first, "fake", "v3", overwrite=False) is False
    assert ds.read_candidates(0)["boxes"] == second


def test_draft_ready_commit_updates_label_and_log(tmp_path):
    ds = dataset(tmp_path)
    boxes = [{"class_id": 7, "x": .5, "y": .5, "width": .2, "height": .2}]
    ds.save_draft(0, boxes)
    ds.set_review(0, "draft_ready")
    preview = ds.commit_preview([0])
    assert preview["count"] == 1
    result = ds.commit_drafts([0])
    assert result["committed"] == 1
    assert ds.read_boxes(0) == boxes
    assert ds.read_draft(0) is None
    assert ds.image_info(0)["status"] == "reviewed"
    assert (tmp_path / ".d1_annotations" / "commit_log.jsonl").exists()


def test_draft_saved_status_is_distinct_from_ready(tmp_path):
    ds = dataset(tmp_path)
    ds.set_review(0, "draft_saved")
    assert ds.image_info(0)["status"] == "draft_saved"
    ds.set_review(0, "draft_ready")
    assert ds.image_info(0)["status"] == "draft_ready"


def test_scan_and_build_with_global_ids(tmp_path):
    from d1_annotation.datasets import DatasetSource, build_dataset, scan_source
    root = tmp_path / "source"
    image = root / "images" / "train" / "a.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (100, 80), "black").save(image)
    label = root / "labels" / "train" / "a.txt"
    label.parent.mkdir(parents=True)
    label.write_text("7 0.5 0.5 0.2 0.2\n8 0.3 0.3 0.1 0.1\n")
    classes = load_classes(CLASSES)
    source = DatasetSource("src", "Source", root)
    scan = scan_source(source, classes)
    assert scan["images"] == 1
    assert scan["classes"]["7"]["images"] == 1
    output = tmp_path / "out"
    report = build_dataset([scan], classes, output, [7, 8], {7: 1, 8: 1}, seed=0)
    assert report["selected_images"] == 1
    labels = list((output / "labels").rglob("*.txt"))
    assert len(labels) == 1
    assert labels[0].read_text().splitlines()[0].startswith("7 ")
    assert (output / "manifest.jsonl").exists()
    assert (output / "statistics.json").exists()


def test_scan_cache_and_dry_run(tmp_path):
    from d1_annotation.datasets import DatasetSource, load_scan_cache, save_scan_cache, scan_source, build_dataset
    root = tmp_path / "source"
    image = root / "images" / "train" / "a.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32), "black").save(image)
    label = root / "labels" / "train" / "a.txt"
    label.parent.mkdir(parents=True)
    label.write_text("7 0.5 0.5 0.2 0.2\n")
    source = DatasetSource("src", "Source", root, "self_collected")
    scan = scan_source(source, load_classes(CLASSES))
    cache_dir = tmp_path / "cache"
    save_scan_cache(cache_dir, scan)
    cached = load_scan_cache(cache_dir, source)
    assert cached["images"] == 1
    report = build_dataset([cached], load_classes(CLASSES), tmp_path / "out", [7], {7: 1}, dry_run=True)
    assert report["dry_run"] is True
    assert report["selected_images"] == 1
    assert not (tmp_path / "out" / "manifest.jsonl").exists()


def test_image_index_cache_avoids_rescan(tmp_path, monkeypatch):
    classes = load_classes(CLASSES)
    image = tmp_path / "images" / "train" / "sample.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (100, 80), "black").save(image)

    first = YoloDataset(tmp_path, classes)
    assert first.image_index_path.exists()
    assert first.list_images()[0]["key"] == "images/train/sample.jpg"
    assert "width" not in first.list_images()[0]

    def fail_scan(_self):
        raise AssertionError("cache should avoid recursive image scan")

    monkeypatch.setattr(YoloDataset, "_scan_images", fail_scan)
    second = YoloDataset(tmp_path, classes)
    assert [item.key for item in second.items] == ["images/train/sample.jpg"]
    assert second.image_info(0)["width"] == 100


def test_rescan_refreshes_image_index(tmp_path):
    classes = load_classes(CLASSES)
    first_image = tmp_path / "images" / "train" / "a.jpg"
    first_image.parent.mkdir(parents=True)
    Image.new("RGB", (10, 10), "black").save(first_image)
    ds = YoloDataset(tmp_path, classes)
    second_image = tmp_path / "images" / "val" / "b.jpg"
    second_image.parent.mkdir(parents=True)
    Image.new("RGB", (20, 20), "black").save(second_image)

    # Directory mtimes provide a cheap invalidation check for newly added files.
    assert len(YoloDataset(tmp_path, classes).items) == 2
    assert ds.rescan_images() == 2
    assert len(YoloDataset(tmp_path, classes).items) == 2


def test_image_page_filters_without_returning_all_items(tmp_path):
    classes = load_classes(CLASSES)
    for index in range(5):
        image = tmp_path / "images" / "train" / f"sample_{index}.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (10, 10), "black").save(image)
    ds = YoloDataset(tmp_path, classes)
    ds.set_review(3, "reviewed")
    page = ds.image_page(page=2, page_size=2)
    assert [row["index"] for row in page["items"]] == [2, 3]
    assert page["total"] == 5
    assert page["pages"] == 3
    assert page["status_counts"]["reviewed"] == 1
    filtered = ds.image_page(query="sample_4", page_size=2)
    assert [row["index"] for row in filtered["items"]] == [4]


def test_scan_reports_empty_abnormal_and_valid_boxes(tmp_path):
    from d1_annotation.datasets import DatasetSource, scan_source
    root = tmp_path / "source"
    image_dir = root / "images" / "train"
    label_dir = root / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for name in ("empty", "mixed", "invalid"):
        Image.new("RGB", (20, 20), "black").save(image_dir / f"{name}.jpg")
    (label_dir / "empty.txt").write_text("\n  \n", encoding="utf-8")
    (label_dir / "mixed.txt").write_text(
        "7 0.5 0.5 0.2 0.2\n7 1.1 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (label_dir / "invalid.txt").write_text("7 0.5 0.5 0.2\n", encoding="utf-8")
    scan = scan_source(DatasetSource("src", "src", root), load_classes(CLASSES))
    q = scan["quality"]
    assert q["physical_empty_labels"] == 1
    assert q["abnormal_boxes"] == 2
    assert q["raw_boxes"] == 3
    assert q["valid_boxes"] == 1
    assert q["no_valid_boxes"] == 2
    assert scan["splits"]["train"]["boxes"] == 1


def test_build_excludes_mixed_invalid_samples(tmp_path):
    from d1_annotation.datasets import DatasetSource, build_dataset, scan_source
    root = tmp_path / "source"
    image_dir = root / "images" / "train"
    label_dir = root / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for name in ("clean", "mixed"):
        Image.new("RGB", (20, 20), "black").save(image_dir / f"{name}.jpg")
    (label_dir / "clean.txt").write_text("7 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (label_dir / "mixed.txt").write_text("7 0.5 0.5 0.2 0.2\n7 1.1 0.5 0.2 0.2\n", encoding="utf-8")
    scan = scan_source(DatasetSource("src", "src", root), load_classes(CLASSES))
    samples = {sample.key: sample for sample in scan["samples"]}
    assert samples["images/train/clean.jpg"].eligible is True
    assert samples["images/train/mixed.jpg"].eligible is False
    report = build_dataset([scan], load_classes(CLASSES), tmp_path / "out", [7], {7: 2}, dry_run=True)
    assert report["eligible_candidates"] == 1
    assert report["selected_images"] == 1
    assert report["available_shortage"]["7"] == 1


def test_build_supports_per_dataset_class_quotas_and_split_ratio(tmp_path):
    from d1_annotation.datasets import DatasetSource, build_dataset, scan_source
    classes = load_classes(CLASSES)
    scans = []
    for source_id, count in (("a", 2), ("b", 3)):
        root = tmp_path / source_id
        (root / "images" / "train").mkdir(parents=True)
        (root / "labels" / "train").mkdir(parents=True)
        for i in range(count):
            Image.new("RGB", (20, 20), "black").save(root / "images" / "train" / f"{i}.jpg")
            (root / "labels" / "train" / f"{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        scans.append(scan_source(DatasetSource(source_id, source_id, root), classes))
    report = build_dataset(
        scans, classes, tmp_path / "out", [0], {}, seed=0,
        dataset_targets={"a": {0: 1}, "b": {0: 2}},
        split_ratio=(0.5, 0.25, 0.25), dry_run=True,
    )
    assert report["selected_images"] == 3
    assert report["selected_by_dataset"] == {"a": {"0": 1}, "b": {"0": 2}}
    assert report["available_shortage"]["a"][0] == 0
    assert report["available_shortage"]["b"][0] == 0
    assert report["split_ratio"] == [0.5, 0.25, 0.25]


def test_scan_distinguishes_pure_and_mixed_abnormal_files(tmp_path):
    from d1_annotation.datasets import DatasetSource, scan_source
    root = tmp_path / "source"
    image_dir = root / "images" / "train"
    label_dir = root / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for name in ("empty", "pure", "mixed", "clean"):
        Image.new("RGB", (20, 20), "black").save(image_dir / f"{name}.jpg")
    (label_dir / "empty.txt").write_text("\n", encoding="utf-8")
    (label_dir / "pure.txt").write_text("7 1.1 0.5 0.2 0.2\n", encoding="utf-8")
    (label_dir / "mixed.txt").write_text("7 0.5 0.5 0.2 0.2\n7 1.1 0.5 0.2 0.2\n", encoding="utf-8")
    (label_dir / "clean.txt").write_text("7 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    scan = scan_source(DatasetSource("src", "src", root), load_classes(CLASSES))
    q = scan["quality"]
    assert q["physical_empty_labels"] == 1
    assert q["pure_abnormal_label_files"] == 1
    assert q["abnormal_label_files"] == 2
    assert q["no_valid_boxes"] == 2
    assert q["valid_boxes"] == 2
