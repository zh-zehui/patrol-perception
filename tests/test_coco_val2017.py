from pathlib import Path

# 轻量：只测 URL / 变体表 / local_dir 重映射，不真正下载
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "download_from_sources",
    ROOT / "datasets/scripts/download_from_sources.py",
)
assert _SPEC and _SPEC.loader
_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_mod)


def test_coco_val2017_variant_registered():
    assert "val2017" in _mod.COCO_ZIP or hasattr(_mod, "COCO_VAL2017") or hasattr(
        _mod, "COCO_2017"
    )
    meta = getattr(_mod, "COCO_2017", None) or getattr(_mod, "COCO_VAL2017", None)
    if meta is None:
        assert "val2017" in _mod.COCO_ZIP
    else:
        images = meta.get("images") or meta.get("val2017")
        labels = meta.get("labels")
        assert "val2017.zip" in str(images) or "val2017.zip" in str(meta.get("val2017", ""))
        assert "coco2017labels" in str(labels)


def test_coco_train2017_variant_registered():
    meta = getattr(_mod, "COCO_2017", None)
    assert meta is not None
    assert "train2017.zip" in meta["train2017"]
    assert "coco2017labels" in meta["labels"]
    assert "train2017" in _mod.COCO_2017_VARIANTS


def test_remap_coco_local_dir_train2017():
    rem = _mod.remap_coco_local_dir
    base = Path("/data/datasets/downloads/coco_val2017")
    assert rem(base, "train2017") == Path("/data/datasets/downloads/coco_train2017")
    assert rem(base, "val2017") == base
    already = Path("/data/datasets/downloads/coco_train2017")
    assert rem(already, "train2017") == already


def test_write_coco_2017_data_yaml_train_split(tmp_path: Path | None = None):
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out = _mod._write_coco_2017_data_yaml(root, "train2017")
        text = out.read_text(encoding="utf-8")
        assert "images/train2017" in text
        assert "val2017" not in text or "train2017" in text
