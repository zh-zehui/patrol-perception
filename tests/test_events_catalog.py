from pathlib import Path

import yaml

from patrol_perception.catalog import (
    apply_enable_overrides,
    apply_self_dir_overrides,
    detect_class_union,
    is_event_inferable,
    iter_enabled_public,
    iter_enabled_self,
    load_catalog,
    load_dataset_sources,
    modules_union,
    validate_catalog,
    validate_dataset_sources,
    validate_detect_data,
)

ROOT = Path(__file__).resolve().parents[1]


def test_all_status_and_enabled_fields():
    data = yaml.safe_load((ROOT / "config/events_catalog.yaml").read_text())
    allowed = {"ship", "partial", "wontfix_phase1"}
    for e in data["events"]:
        assert e["status"] in allowed
        assert "enabled" in e and "depends_on" in e
        if e["status"] in ("partial", "wontfix_phase1"):
            assert e.get("reason")


def test_every_detect_class_has_data_tier():
    validate_detect_data()


def test_dataset_sources_urls_and_dirs():
    validate_dataset_sources()
    src = load_dataset_sources(ROOT / "config/dataset_sources.yaml")
    pubs = iter_enabled_public(src, "smoke")
    assert pubs and pubs[0]["url"].startswith("http")
    # The template documents a local source but leaves absent datasets disabled.
    selves = src["classes"]["leash"]["self"]
    assert selves and selves[0]["dir"]
    assert iter_enabled_self(src, "leash") == [s for s in selves if s.get("enabled")]


def test_self_dir_override():
    src = load_dataset_sources(ROOT / "config/dataset_sources.yaml")
    src2 = apply_self_dir_overrides(src, {"ebike": "/data/my_ebike"})
    assert iter_enabled_self(src2, "ebike")[0]["dir"] == "/data/my_ebike"


def test_no_motorcycle_to_ebike_alias():
    cfg = yaml.safe_load((ROOT / "config/classes.yaml").read_text())
    assert cfg.get("coco_alias", {}).get("motorcycle") is None
    assert cfg["detect_data"]["ebike"]["data_tier"] == "public_online"
    assert cfg["detect_data"]["litter_bottle"]["data_tier"] == "coco_approx"
    assert cfg["detect_data"]["litter_bottle"]["coco_source"] == "bottle"


def test_validate_catalog_ok():
    cat = load_catalog(ROOT / "config/events_catalog.yaml")
    validate_catalog(cat)


def test_wontfix_not_inferable():
    cat = load_catalog(ROOT / "config/events_catalog.yaml")
    assert is_event_inferable(cat, "curb_step") is False
    assert is_event_inferable(cat, "trash_overflow") is False


def test_fence_climb_partial_wired_to_detect_classes():
    cat = load_catalog(ROOT / "config/events_catalog.yaml")
    ev = next(e for e in cat["events"] if e["id"] == "fence_climb")
    assert ev["status"] == "partial"
    assert set(ev["depends_on"]["detect"]) >= {"person", "fence", "climbing"}
    # 默认关，使能后才可推理
    assert is_event_inferable(cat, "fence_climb") is False
    cat2 = apply_enable_overrides(cat, enable=["fence_climb"])
    assert is_event_inferable(cat2, "fence_climb") is True


def test_disable_smoking_drops_cigarette_when_unused():
    cat = load_catalog(ROOT / "config/events_catalog.yaml")
    cat = apply_enable_overrides(cat, enable=None, disable=["smoking"])
    assert "cigarette_or_smoke" not in detect_class_union(cat)


def test_disable_bicycle_keeps_class_if_mispark_enabled():
    cat = load_catalog(ROOT / "config/events_catalog.yaml")
    cat = apply_enable_overrides(cat, enable=None, disable=["bicycle"])
    # micromobility_mispark / nav still need bicycle
    assert "bicycle" in detect_class_union(cat)


def test_modules_include_face_when_face_capture_on():
    cat = load_catalog(ROOT / "config/events_catalog.yaml")
    assert "face" in modules_union(cat)
