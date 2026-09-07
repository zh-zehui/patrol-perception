"""summarize_yolo_dataset — 统计 YOLO 下载/合并集各类 box / 含类图数。"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_mod():
    path = ROOT / "datasets/scripts/summarize_yolo_dataset.py"
    spec = importlib.util.spec_from_file_location("summarize_yolo_dataset", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestSummarizeYoloDataset(unittest.TestCase):
    def test_standard_train_val_layout(self):
        mod = _load_mod()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split, labels in (
                ("train", {"a": "0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1\n", "b": "0 0.1 0.1 0.2 0.2\n"}),
                ("val", {"c": "1 0.5 0.5 0.1 0.1\n"}),
            ):
                (root / "images" / split).mkdir(parents=True)
                (root / "labels" / split).mkdir(parents=True)
                for stem, text in labels.items():
                    (root / "images" / split / f"{stem}.jpg").write_bytes(b"x")
                    (root / "labels" / split / f"{stem}.txt").write_text(text, encoding="utf-8")
            (root / "data.yaml").write_text(
                yaml.safe_dump(
                    {
                        "path": str(root),
                        "train": "images/train",
                        "val": "images/val",
                        "names": {0: "person", 1: "bicycle"},
                    }
                ),
                encoding="utf-8",
            )

            report = mod.summarize_dataset(root)
            self.assertEqual(report["names"], ["person", "bicycle"])
            train = report["splits"]["train"]
            self.assertEqual(train["label_files"], 2)
            self.assertEqual(train["boxes_by_class"]["person"], 2)
            self.assertEqual(train["boxes_by_class"]["bicycle"], 1)
            self.assertEqual(train["images_by_class"]["person"], 2)
            self.assertEqual(train["images_by_class"]["bicycle"], 1)
            val = report["splits"]["val"]
            self.assertEqual(val["boxes_by_class"]["bicycle"], 1)
            self.assertEqual(val["boxes_by_class"].get("person", 0), 0)

    def test_coco_style_val2017_subdir_and_class_filter(self):
        mod = _load_mod()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lab = root / "labels" / "val2017"
            img = root / "images" / "val2017"
            lab.mkdir(parents=True)
            img.mkdir(parents=True)
            (lab / "x.txt").write_text(
                "0 0.5 0.5 0.1 0.1\n7 0.2 0.2 0.1 0.1\n1 0.3 0.3 0.1 0.1\n",
                encoding="utf-8",
            )
            (img / "x.jpg").write_bytes(b"x")
            (root / "data.yaml").write_text(
                yaml.safe_dump({"names": {0: "person", 1: "bicycle", 7: "truck"}}),
                encoding="utf-8",
            )

            report = mod.summarize_dataset(
                root,
                class_ids={0, 1, 7},
                names_override={0: "person", 1: "bicycle", 7: "truck"},
            )
            split = report["splits"]["val2017"]
            self.assertEqual(split["boxes_by_class"]["person"], 1)
            self.assertEqual(split["boxes_by_class"]["bicycle"], 1)
            self.assertEqual(split["boxes_by_class"]["truck"], 1)


if __name__ == "__main__":
    unittest.main()
