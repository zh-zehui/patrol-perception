from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from patrol_perception.eval.metrics import (
    build_metrics_doc,
    empty_class_metrics,
    extract_from_ultralytics,
)


class TestEmptyClassMetrics(unittest.TestCase):
    def test_empty_class_metrics_shape(self):
        m = empty_class_metrics()
        self.assertEqual(
            set(m.keys()),
            {"ap50", "ap50_95", "precision", "recall"},
        )
        self.assertIsNone(m["ap50"])
        self.assertIsNone(m["ap50_95"])


class TestBuildMetricsDoc(unittest.TestCase):
    def test_build_metrics_doc_shape(self):
        per_class = {
            "person": {"ap50": 0.8, "ap50_95": 0.5, "precision": 0.7, "recall": 0.6},
        }
        summary = {"map50": 0.8, "map50_95": 0.5}
        doc = build_metrics_doc(
            per_class,
            summary,
            split="val",
            names=["person"],
        )
        self.assertEqual(doc["split"], "val")
        self.assertEqual(doc["names"], ["person"])
        self.assertEqual(doc["per_class"], per_class)
        self.assertEqual(doc["summary"], summary)


class TestExtract(unittest.TestCase):
    def test_extract_maps_box_metrics(self):
        # 模拟 Ultralytics DetMetrics：maps 为每类 AP50-95；ap50 分列
        names = ["person", "smoke"]
        box = SimpleNamespace(
            maps=[0.55, 0.25],  # AP50-95 per class
            ap50=[0.80, 0.40],
            mp=0.7,
            mr=0.6,
            map50=0.6,
            map=0.4,
            p=[0.75, 0.5],
            r=[0.7, 0.45],
        )
        results = SimpleNamespace(box=box)
        doc = extract_from_ultralytics(results, names)
        self.assertEqual(doc["per_class"]["person"]["ap50"], 0.80)
        self.assertEqual(doc["per_class"]["smoke"]["ap50_95"], 0.25)
        self.assertEqual(doc["summary"]["map50"], 0.6)
        self.assertEqual(doc["summary"]["map50_95"], 0.4)
        self.assertEqual(doc["split"], "val")
        self.assertEqual(doc["names"], names)
        self.assertEqual(doc["per_class"]["person"]["precision"], 0.75)
        self.assertEqual(doc["per_class"]["smoke"]["recall"], 0.45)

    def test_extract_ap_class_index_sparse_classes(self):
        # Ultralytics 真实 val：仅有 GT 的类出现在 ap_class_index；ap50/p/r 按该顺序排列
        names = ["person", "smoke", "fire"]
        box = SimpleNamespace(
            maps=[0.55, 0.30, 0.20],
            ap_class_index=[1],
            ap50=[0.72],
            p=[0.68],
            r=[0.65],
            map50=0.72,
            map=0.35,
        )
        results = SimpleNamespace(box=box)
        doc = extract_from_ultralytics(results, names)

        person = doc["per_class"]["person"]
        smoke = doc["per_class"]["smoke"]
        fire = doc["per_class"]["fire"]

        # person/fire have no GT (absent from ap_class_index): ap50_95 must
        # be None just like ap50/precision/recall, not read from maps[i].
        self.assertIsNone(person["ap50"])
        self.assertIsNone(person["precision"])
        self.assertIsNone(person["recall"])
        self.assertIsNone(person["ap50_95"])

        self.assertEqual(smoke["ap50"], 0.72)
        self.assertEqual(smoke["precision"], 0.68)
        self.assertEqual(smoke["recall"], 0.65)
        self.assertEqual(smoke["ap50_95"], 0.30)

        self.assertIsNone(fire["ap50"])
        self.assertIsNone(fire["precision"])
        self.assertIsNone(fire["recall"])
        self.assertIsNone(fire["ap50_95"])

        self.assertEqual(doc["summary"]["map50"], 0.72)
        self.assertEqual(doc["summary"]["map50_95"], 0.35)

    def test_extract_ap_class_index_absent_class_maps_filler_ignored(self):
        # 复现真实 Ultralytics 行为：不在 ap_class_index 中的类，maps[i] 被填充为
        # 整体 mAP（哨兵值），而非该类真实的 AP50-95；这些类必须得到 ap50_95=None，
        # 就像 ap50/precision/recall 一样，不能读到这个哨兵值。
        names = ["person", "smoke"]
        box = SimpleNamespace(
            maps=[0.0005, 0.25],  # maps[0] 是全局 mAP 哨兵值（person 无 GT）
            ap_class_index=[1],
            ap50=[0.68],
            p=[0.6],
            r=[0.55],
            map50=0.68,
            map=0.25,
        )
        results = SimpleNamespace(box=box)
        doc = extract_from_ultralytics(results, names, split="test")

        person = doc["per_class"]["person"]
        smoke = doc["per_class"]["smoke"]

        self.assertIsNone(person["ap50"])
        self.assertIsNone(person["precision"])
        self.assertIsNone(person["recall"])
        self.assertIsNone(person["ap50_95"])

        self.assertEqual(smoke["ap50"], 0.68)
        self.assertEqual(smoke["precision"], 0.6)
        self.assertEqual(smoke["recall"], 0.55)
        self.assertEqual(smoke["ap50_95"], 0.25)


if __name__ == "__main__":
    unittest.main()
