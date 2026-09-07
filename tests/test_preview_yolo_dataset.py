"""preview_yolo_dataset：按类/prefix 抽图，各出 raw + labeled 两张拼图。"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def _load_mod():
    path = ROOT / "datasets/scripts/preview_yolo_dataset.py"
    spec = importlib.util.spec_from_file_location("preview_yolo_dataset", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_merged_set(root: Path) -> Path:
    """人造 detect 联合集：两个 prefix，两类。"""
    names = {0: "person", 1: "ebike"}
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
    # coco-like person
    for i in range(6):
        im = Image.new("RGB", (80, 60), (200, 200, 200))
        for x in range(10, 40):
            for y in range(10, 40):
                im.putpixel((x, y), (20, 20, 200))
        stem = f"coco_p{i}"
        im.save(root / "images" / "val" / f"{stem}.jpg")
        (root / "labels" / "val" / f"{stem}.txt").write_text(
            "0 0.3 0.4 0.3 0.4\n", encoding="utf-8"
        )
    # ebike prefix
    for i in range(6):
        im = Image.new("RGB", (80, 60), (240, 240, 240))
        for x in range(20, 70):
            for y in range(15, 50):
                im.putpixel((x, y), (30, 30, 30))
        stem = f"ebike_e{i}"
        im.save(root / "images" / "val" / f"{stem}.jpg")
        (root / "labels" / "val" / f"{stem}.txt").write_text(
            "1 0.55 0.55 0.5 0.5\n", encoding="utf-8"
        )
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "names": names,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return root


class TestPreviewYoloDataset(unittest.TestCase):
    def test_make_contact_sheet_pair_writes_raw_and_labeled(self):
        mod = _load_mod()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            root = _make_merged_set(td / "detect")
            out = td / "preview"
            meta = mod.preview_dataset(
                root,
                out,
                split="val",
                per_group=4,
                seed=0,
                modes=("class", "prefix"),
                cell_size=64,
            )
            # 按类：person / ebike 各一对
            self.assertTrue((out / "by_class" / "person_raw.jpg").is_file())
            self.assertTrue((out / "by_class" / "person_labeled.jpg").is_file())
            self.assertTrue((out / "by_class" / "ebike_raw.jpg").is_file())
            self.assertTrue((out / "by_class" / "ebike_labeled.jpg").is_file())
            # 按 prefix
            self.assertTrue((out / "by_prefix" / "coco_raw.jpg").is_file())
            self.assertTrue((out / "by_prefix" / "coco_labeled.jpg").is_file())
            self.assertTrue((out / "by_prefix" / "ebike_raw.jpg").is_file())
            self.assertTrue((out / "by_prefix" / "ebike_labeled.jpg").is_file())
            self.assertIn("groups", meta)
            # raw 与 labeled 尺寸应一致
            with Image.open(out / "by_class" / "person_raw.jpg") as raw:
                with Image.open(out / "by_class" / "person_labeled.jpg") as lab:
                    self.assertEqual(raw.size, lab.size)

    def test_prefix_from_filename(self):
        mod = _load_mod()
        self.assertEqual(mod.prefix_from_stem("cig2_smoke_3331"), "cig2")
        self.assertEqual(mod.prefix_from_stem("coco_0001"), "coco")
        self.assertEqual(mod.prefix_from_stem("nounderscore"), "nounderscore")
        known = ["coco", "coco_pet", "cig", "cig2"]
        self.assertEqual(mod.prefix_from_stem("coco_pet_000000357663", known), "coco_pet")
        self.assertEqual(mod.prefix_from_stem("coco_000000123", known), "coco")
        self.assertEqual(mod.prefix_from_stem("cig2_0", known), "cig2")
        self.assertEqual(mod.prefix_from_stem("cig_0", known), "cig")
        self.assertTrue(mod.stem_matches_prefix("coco_pet_0001", "coco_pet"))
        self.assertFalse(mod.stem_matches_prefix("coco_0001", "coco_pet"))

    def test_prefixes_for_classes_from_merge_meta(self):
        mod = _load_mod()
        meta = {
            "sets": [
                {"prefix": "cig", "keep": ["cigarette_or_smoke"]},
                {"prefix": "cig2", "keep": ["cigarette_or_smoke"]},
                {"prefix": "taco", "keep": ["litter_bottle", "litter_paper"]},
                {"prefix": "fire", "keep": ["flame", "smoke"]},
            ]
        }
        m = mod.prefixes_for_classes(meta, ["cigarette_or_smoke"])
        self.assertEqual(m["cigarette_or_smoke"], ["cig", "cig2"])
        m2 = mod.prefixes_for_classes(meta, ["litter_bottle", "flame"])
        self.assertEqual(m2["litter_bottle"], ["taco"])
        self.assertEqual(m2["flame"], ["fire"])

    def test_parse_menu_choice(self):
        mod = _load_mod()
        opts = ["cig", "cig2", "taco"]
        self.assertEqual(mod.parse_menu_choice("1,3", opts), ["cig", "taco"])
        self.assertEqual(mod.parse_menu_choice("a", opts), opts)
        self.assertEqual(mod.parse_menu_choice("all", opts), opts)
        self.assertEqual(mod.parse_menu_choice("2", opts), ["cig2"])

    def test_resolve_prefixes_interactive_from_merge_meta(self):
        mod = _load_mod()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "merge_meta.json").write_text(
                '{"sets":[{"prefix":"cig","keep":["cigarette_or_smoke"]},'
                '{"prefix":"cig2","keep":["cigarette_or_smoke"]}]}',
                encoding="utf-8",
            )
            got = mod.resolve_prefixes_interactive(
                td,
                ["cigarette_or_smoke"],
                None,
                yes=False,
                input_fn=lambda _p: "2",
            )
            self.assertEqual(got, ["cig2"])
            got_all = mod.resolve_prefixes_interactive(
                td, ["cigarette_or_smoke"], None, yes=True
            )
            self.assertEqual(got_all, ["cig", "cig2"])

    def test_class_preview_respects_selected_prefixes(self):
        """抽 cigarette 类时若只选 cig2，不应混入 cig_ 文件。"""
        mod = _load_mod()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            root = td / "detect"
            names = {0: "cigarette_or_smoke"}
            (root / "images" / "val").mkdir(parents=True)
            (root / "labels" / "val").mkdir(parents=True)
            for pref, n in (("cig", 5), ("cig2", 5)):
                for i in range(n):
                    im = Image.new("RGB", (40, 40), (10, 10, 10))
                    stem = f"{pref}_{i}"
                    im.save(root / "images" / "val" / f"{stem}.jpg")
                    (root / "labels" / "val" / f"{stem}.txt").write_text(
                        "0 0.5 0.5 0.4 0.4\n", encoding="utf-8"
                    )
            (root / "data.yaml").write_text(
                yaml.safe_dump(
                    {"path": str(root), "val": "images/val", "names": names},
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            (root / "merge_meta.json").write_text(
                '{"sets":[{"prefix":"cig","keep":["cigarette_or_smoke"]},'
                '{"prefix":"cig2","keep":["cigarette_or_smoke"]}]}',
                encoding="utf-8",
            )
            out = td / "preview"
            mod.preview_dataset(
                root,
                out,
                split="val",
                per_group=9,
                seed=0,
                modes=("class",),
                classes=["cigarette_or_smoke"],
                prefixes=["cig2"],
                cell_size=48,
            )
            import json

            pm = json.loads((out / "preview_meta.json").read_text(encoding="utf-8"))
            files = pm["groups"][0]["files"]
            self.assertTrue(all(f.startswith("cig2_") for f in files))
            self.assertFalse(any(f.startswith("cig_") for f in files))

    def test_class_preview_matches_compound_prefix(self):
        """coco_pet 含下划线时，不得被拆成 coco 后抽到 0 张。"""
        mod = _load_mod()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            root = td / "detect"
            names = {0: "person", 1: "pet"}
            (root / "images" / "val").mkdir(parents=True)
            (root / "labels" / "val").mkdir(parents=True)
            for i in range(4):
                Image.new("RGB", (40, 40), (200, 80, 80)).save(
                    root / "images" / "val" / f"coco_{i}.jpg"
                )
                (root / "labels" / "val" / f"coco_{i}.txt").write_text(
                    "0 0.5 0.5 0.4 0.4\n", encoding="utf-8"
                )
            for i in range(4):
                Image.new("RGB", (40, 40), (80, 200, 80)).save(
                    root / "images" / "val" / f"coco_pet_{i}.jpg"
                )
                (root / "labels" / "val" / f"coco_pet_{i}.txt").write_text(
                    "1 0.5 0.5 0.4 0.4\n", encoding="utf-8"
                )
            (root / "data.yaml").write_text(
                yaml.safe_dump(
                    {"path": str(root), "val": "images/val", "names": names},
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            (root / "merge_meta.json").write_text(
                '{"sets":[{"prefix":"coco","keep":["person"]},'
                '{"prefix":"coco_pet","keep":["pet"]}]}',
                encoding="utf-8",
            )
            out = td / "preview"
            meta = mod.preview_dataset(
                root,
                out,
                split="val",
                per_group=9,
                seed=0,
                modes=("class",),
                classes=["pet"],
                prefixes=["coco_pet"],
                cell_size=48,
            )
            files = meta["groups"][0]["files"]
            self.assertEqual(meta["groups"][0]["available"], 4)
            self.assertTrue(files)
            self.assertTrue(all(f.startswith("coco_pet_") for f in files))
            self.assertFalse(any(f.startswith("coco_") and not f.startswith("coco_pet_") for f in files))
            with Image.open(out / "by_class" / "pet__coco_pet_raw.jpg") as im:
                # 有图时不应是单格深灰占位
                self.assertNotEqual(im.size, (48, 48))


if __name__ == "__main__":
    unittest.main()
