# tests/test_eval_compare.py
import unittest
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from patrol_perception.eval.compare import (
    compute_delta,
    resolve_baseline,
    summarize_metrics,
)

class TestEvalCompare(unittest.TestCase):
    def test_summarize_focus_and_base(self):
        per = {
            "person": {"ap50": 0.8, "ap50_95": 0.5},
            "smoke": {"ap50": 0.4, "ap50_95": 0.2},
            "flame": {"ap50": 0.6, "ap50_95": 0.3},
        }
        s = summarize_metrics(per, focus=["smoke", "flame"], base=["person"])
        self.assertAlmostEqual(s["focus_map50"], 0.5)
        self.assertAlmostEqual(s["base_map50"], 0.8)
        self.assertAlmostEqual(s["map50"], (0.8 + 0.4 + 0.6) / 3)

    def test_delta_intersection_only(self):
        cur = {"per_class": {"person": {"ap50": 0.7, "ap50_95": 0.4}, "smoke": {"ap50": 0.5, "ap50_95": 0.2}}}
        base = {"per_class": {"person": {"ap50": 0.9, "ap50_95": 0.6}, "car": {"ap50": 0.8, "ap50_95": 0.5}}}
        d = compute_delta(cur, base, base_classes=["person", "car"])
        self.assertAlmostEqual(d["delta_ap50"]["person"], -0.2)
        self.assertNotIn("smoke", d["delta_ap50"])  # 基线无此类
        self.assertNotIn("car", d["delta_ap50"])    # 当前无此类
        self.assertAlmostEqual(d["delta_base_map50"], -0.2)

    def test_resolve_baseline_cli_overrides_yaml(self):
        cfg = {"baseline": {"default": "data/weights/yolo26l.pt"}}
        self.assertEqual(
            resolve_baseline("data/weights/custom.pt", cfg, no_baseline=False),
            "data/weights/custom.pt",
        )

    def test_resolve_baseline_yaml_when_cli_omitted(self):
        cfg = {"baseline": {"default": "data/weights/yolo26l.pt"}}
        self.assertEqual(
            resolve_baseline(None, cfg, no_baseline=False),
            "data/weights/yolo26l.pt",
        )

    def test_resolve_baseline_no_baseline_skips_yaml(self):
        cfg = {"baseline": {"default": "data/weights/yolo26l.pt"}}
        self.assertIsNone(resolve_baseline(None, cfg, no_baseline=True))
        self.assertIsNone(resolve_baseline("data/weights/custom.pt", cfg, no_baseline=True))

    def test_resolve_baseline_missing_or_empty_yaml(self):
        self.assertIsNone(resolve_baseline(None, {}, no_baseline=False))
        self.assertIsNone(resolve_baseline(None, {"baseline": {}}, no_baseline=False))
        self.assertIsNone(resolve_baseline(None, {"baseline": {"default": ""}}, no_baseline=False))

if __name__ == "__main__":
    unittest.main()
