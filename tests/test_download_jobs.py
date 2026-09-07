from pathlib import Path

from patrol_perception.catalog import (
    collect_public_download_jobs,
    get_data_root,
    load_dataset_sources,
    resolve_data_path,
)

ROOT = Path(__file__).resolve().parents[1]


def test_coco_public_enabled_for_base_classes():
    src = load_dataset_sources(ROOT / "config/dataset_sources.yaml")
    for cls in ("person", "bicycle", "car", "truck"):
        pubs = src["classes"][cls]["public"]
        coco = [p for p in pubs if p.get("enabled") and "coco" in (p.get("local_dir") or "")]
        assert coco, f"{cls} should have enabled COCO public"
        assert coco[0]["local_dir"] == "datasets/downloads/coco_val2017"
        assert coco[0].get("kind") == "coco"


def test_public_jobs_dedupe_by_url_and_local_dir():
    src = load_dataset_sources(ROOT / "config/dataset_sources.yaml")
    jobs = collect_public_download_jobs(
        src,
        ["person", "bicycle", "car", "truck"],
        root=ROOT,
    )
    assert len(jobs) == 1
    job = jobs[0]
    assert set(job["classes"]) == {"person", "bicycle", "car", "truck"}
    assert job["kind"] == "coco"
    assert job["local_dir"] == (ROOT / "datasets/downloads/coco_val2017").resolve()
    assert "docs.ultralytics.com/datasets/detect/coco" in job["url"]


def test_dedupe_keeps_distinct_dirs():
    sources = {
        "classes": {
            "a": {
                "public": [
                    {
                        "name": "x",
                        "url": "https://example.com/a.zip",
                        "local_dir": "datasets/downloads/a",
                        "enabled": True,
                    }
                ]
            },
            "b": {
                "public": [
                    {
                        "name": "y",
                        "url": "https://example.com/a.zip",
                        "local_dir": "datasets/downloads/b",
                        "enabled": True,
                    }
                ]
            },
            "c": {
                "public": [
                    {
                        "name": "z",
                        "url": "https://example.com/a.zip",
                        "local_dir": "datasets/downloads/a",
                        "enabled": True,
                    }
                ]
            },
        }
    }
    jobs = collect_public_download_jobs(sources, root=ROOT)
    assert len(jobs) == 2
    by_dir = {j["local_dir"]: j for j in jobs}
    assert set(by_dir[(ROOT / "datasets/downloads/a").resolve()]["classes"]) == {"a", "c"}
    assert by_dir[(ROOT / "datasets/downloads/b").resolve()]["classes"] == ["b"]


def test_data_root_absolute_prefixes_relative_local_dir():
    sources = {
        "roots": {"data_root": "/mnt/data_hdd/patrol_perception"},
        "classes": {
            "person": {
                "public": [
                    {
                        "name": "coco",
                        "url": "https://docs.ultralytics.com/datasets/detect/coco/",
                        "local_dir": "datasets/downloads/coco",
                        "enabled": True,
                        "kind": "coco",
                    }
                ]
            }
        },
    }
    jobs = collect_public_download_jobs(sources, root=ROOT)
    assert len(jobs) == 1
    assert jobs[0]["local_dir"] == Path(
        "/mnt/data_hdd/patrol_perception/datasets/downloads/coco"
    ).resolve()


def test_absolute_local_dir_ignores_data_root():
    sources = {
        "roots": {"data_root": "/mnt/data_hdd/patrol_perception"},
        "classes": {
            "person": {
                "public": [
                    {
                        "name": "coco",
                        "url": "https://example.com/coco.zip",
                        "local_dir": "/media/ext/coco_only",
                        "enabled": True,
                    }
                ]
            }
        },
    }
    jobs = collect_public_download_jobs(sources, root=ROOT)
    assert jobs[0]["local_dir"] == Path("/media/ext/coco_only").resolve()


def test_cli_data_root_override():
    sources = {
        "roots": {"data_root": "/mnt/old"},
        "classes": {
            "x": {
                "public": [
                    {
                        "name": "a",
                        "url": "https://example.com/a.zip",
                        "local_dir": "datasets/downloads/a",
                        "enabled": True,
                    }
                ]
            }
        },
    }
    jobs = collect_public_download_jobs(
        sources, root=ROOT, data_root="/mnt/new_hdd/patrol"
    )
    assert jobs[0]["local_dir"] == Path("/mnt/new_hdd/patrol/datasets/downloads/a").resolve()
    assert get_data_root(sources, project_root=ROOT, override="/tmp/ovr") == Path(
        "/tmp/ovr"
    ).resolve()
    assert resolve_data_path(
        "datasets/downloads/coco",
        {"roots": {"data_root": "/mnt/hdd"}},
        project_root=ROOT,
    ) == Path("/mnt/hdd/datasets/downloads/coco").resolve()


def test_hf_kind_auto_and_repo_id():
    import importlib.util

    path = ROOT / "datasets/scripts/download_from_sources.py"
    spec = importlib.util.spec_from_file_location("download_from_sources", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    url = "https://huggingface.co/datasets/jojomoi-meme/fog_fire_detection"
    assert mod.parse_hf_dataset_repo_id(url) == "jojomoi-meme/fog_fire_detection"

    sources = {
        "classes": {
            "smoke": {
                "public": [
                    {
                        "name": "hf",
                        "url": url,
                        "local_dir": "datasets/downloads/fog_fire_detection",
                        "enabled": True,
                    }
                ]
            }
        }
    }
    jobs = collect_public_download_jobs(sources, root=ROOT)
    assert len(jobs) == 1
    assert jobs[0]["kind"] == "huggingface"


def test_ndjson_job_passes_path():
    sources = {
        "classes": {
            "cigarette_or_smoke": {
                "public": [
                    {
                        "name": "cig",
                        "url": "https://universe.roboflow.com/liu-xianglei/cigarette",
                        "local_dir": "datasets/downloads/cigarette_roboflow",
                        "enabled": True,
                        "kind": "ndjson",
                        "ndjson": "~/Downloads/cigarette.ndjson",
                    }
                ]
            }
        }
    }
    jobs = collect_public_download_jobs(sources, root=ROOT)
    assert len(jobs) == 1
    assert jobs[0]["kind"] == "ndjson"
    assert jobs[0]["ndjson"] == "~/Downloads/cigarette.ndjson"

    import importlib.util

    path = ROOT / "datasets/scripts/download_from_sources.py"
    spec = importlib.util.spec_from_file_location("download_from_sources", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    resolved = mod.resolve_ndjson_path("~/Downloads/cigarette.ndjson")
    assert resolved is None or resolved.name == "cigarette.ndjson"


def test_taco_jobs_dedupe():
    src = load_dataset_sources(ROOT / "config/dataset_sources.yaml")
    jobs = collect_public_download_jobs(
        src,
        ["litter_bottle", "litter_plastic_bag", "litter_paper", "litter_cardboard"],
        root=ROOT,
    )
    taco = [j for j in jobs if j.get("kind") == "taco"]
    assert len(taco) == 1
    assert set(taco[0]["classes"]) == {
        "litter_bottle",
        "litter_plastic_bag",
        "litter_paper",
        "litter_cardboard",
    }
    assert "pedropro/TACO" in taco[0]["url"] or "taco" in taco[0]["url"].lower()
