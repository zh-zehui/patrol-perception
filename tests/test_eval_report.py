import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from patrol_perception.eval.report import write_eval_run


class TestEvalReport(unittest.TestCase):
    def test_write_eval_run_creates_files(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            out = td / "w1_fire_20260101_000000"
            index = td / "index.jsonl"
            meta = {"wave": "w1_fire", "weights": "best.pt", "data": "data.yaml"}
            metrics_val = {
                "split": "val",
                "names": ["person", "smoke"],
                "per_class": {
                    "person": {
                        "ap50": 0.8,
                        "ap50_95": 0.5,
                        "precision": 0.7,
                        "recall": 0.6,
                    },
                    "smoke": {
                        "ap50": 0.4,
                        "ap50_95": 0.2,
                        "precision": 0.5,
                        "recall": 0.4,
                    },
                },
                "summary": {
                    "map50": 0.6,
                    "map50_95": 0.35,
                    "focus_map50": 0.4,
                    "base_map50": 0.8,
                },
            }
            write_eval_run(
                out,
                meta=meta,
                metrics_by_split={"val": metrics_val},
                baseline_val=None,
                delta=None,
                index_path=index,
            )
            self.assertTrue((out / "run_meta.json").is_file())
            self.assertTrue((out / "metrics_val.json").is_file())
            self.assertTrue((out / "report.md").is_file())
            self.assertTrue(index.is_file())
            report = (out / "report.md").read_text()
            self.assertIn("w1_fire", report)
            self.assertIn("Focus", report)
            self.assertIn("Base", report)
            self.assertIn("Summary", report)

            index_line = json.loads(index.read_text().strip().splitlines()[-1])
            self.assertEqual(index_line["wave"], "w1_fire")
            self.assertEqual(index_line["weights"], "best.pt")
            self.assertAlmostEqual(index_line["map50_val"], 0.6)
            self.assertAlmostEqual(index_line["focus_map50"], 0.4)
            self.assertIsNone(index_line["delta_base_map50"])

    def test_write_eval_run_baseline_and_delta(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            out = td / "w1_fire_20260101_000001"
            index = td / "index.jsonl"
            meta = {"wave": "w1_fire", "weights": "best.pt", "data": "data.yaml"}
            metrics_val = {
                "split": "val",
                "names": ["person", "smoke"],
                "per_class": {
                    "person": {"ap50": 0.7, "ap50_95": 0.4, "precision": None, "recall": None},
                    "smoke": {"ap50": 0.5, "ap50_95": 0.25, "precision": None, "recall": None},
                },
                "summary": {
                    "map50": 0.6,
                    "map50_95": 0.325,
                    "focus_map50": 0.5,
                    "base_map50": 0.7,
                },
            }
            baseline_val = {
                "split": "val",
                "names": ["person", "smoke"],
                "per_class": {
                    "person": {"ap50": 0.9, "ap50_95": 0.6, "precision": None, "recall": None},
                    "smoke": {"ap50": 0.3, "ap50_95": 0.15, "precision": None, "recall": None},
                },
                "summary": {"map50": 0.6, "map50_95": 0.375},
            }
            delta = {
                "delta_ap50": {"person": -0.2, "smoke": 0.2},
                "delta_ap50_95": {"person": -0.2, "smoke": 0.1},
                "delta_base_map50": -0.2,
                "compared_classes": ["person", "smoke"],
            }
            write_eval_run(
                out,
                meta=meta,
                metrics_by_split={"val": metrics_val},
                baseline_val=baseline_val,
                delta=delta,
                index_path=index,
            )
            self.assertTrue((out / "metrics_baseline_val.json").is_file())
            self.assertTrue((out / "baseline_delta.json").is_file())
            report = (out / "report.md").read_text()
            self.assertIn("-0.2", report)
            index_line = json.loads(index.read_text().strip().splitlines()[-1])
            self.assertAlmostEqual(index_line["delta_base_map50"], -0.2)

    def test_write_eval_run_renders_test_split(self):
        # Critical #2: report.md must show every split (at least val and
        # test), not just val — spec §7 says test is authoritative for
        # final fire-class numbers.
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            out = td / "w1_fire_20260101_000002"
            index = td / "index.jsonl"
            meta = {
                "wave": "w1_fire",
                "weights": "best.pt",
                "data": "data.yaml",
                "focus_classes": ["smoke"],
                "base_classes": ["person"],
                "wave_known": True,
            }
            metrics_val = {
                "split": "val",
                "names": ["person", "smoke"],
                "per_class": {
                    "person": {"ap50": 0.8, "ap50_95": 0.5, "precision": 0.7, "recall": 0.6},
                    "smoke": {"ap50": 0.4, "ap50_95": 0.2, "precision": 0.5, "recall": 0.4},
                },
                "summary": {"map50": 0.6, "map50_95": 0.35, "focus_map50": 0.4, "base_map50": 0.8},
            }
            metrics_test = {
                "split": "test",
                "names": ["person", "smoke"],
                "per_class": {
                    "person": {"ap50": None, "ap50_95": None, "precision": None, "recall": None},
                    "smoke": {
                        "ap50": 0.1234,
                        "ap50_95": 0.0987,
                        "precision": 0.2,
                        "recall": 0.3,
                    },
                },
                "summary": {
                    "map50": 0.1234,
                    "map50_95": 0.0987,
                    "focus_map50": 0.1234,
                    "base_map50": None,
                },
            }
            write_eval_run(
                out,
                meta=meta,
                metrics_by_split={"val": metrics_val, "test": metrics_test},
                baseline_val=None,
                delta=None,
                index_path=index,
            )
            self.assertTrue((out / "metrics_test.json").is_file())
            report = (out / "report.md").read_text()
            self.assertIn("Split: val", report)
            self.assertIn("Split: test", report)
            # test-split's distinctive smoke ap50_95 value must actually appear
            self.assertIn("0.0987", report)
            # test-split section must come after the val-split section
            self.assertLess(report.index("Split: val"), report.index("Split: test"))

    def test_write_eval_run_unknown_wave_focus_messaging(self):
        # Important: unknown --wave must not silently dump all classes into
        # the Focus section; it must show an explicit "(none — unknown
        # wave ...)" message instead.
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            out = td / "w1_bogus_20260101_000000"
            index = td / "index.jsonl"
            meta = {
                "wave": "w1_bogus",
                "weights": "best.pt",
                "data": "data.yaml",
                "focus_classes": [],
                "base_classes": ["person"],
                "wave_known": False,
            }
            metrics_val = {
                "split": "val",
                "names": ["person", "smoke"],
                "per_class": {
                    "person": {"ap50": 0.8, "ap50_95": 0.5, "precision": 0.7, "recall": 0.6},
                    "smoke": {"ap50": 0.4, "ap50_95": 0.2, "precision": 0.5, "recall": 0.4},
                },
                "summary": {"map50": 0.6, "map50_95": 0.35, "focus_map50": None, "base_map50": 0.8},
            }
            write_eval_run(
                out,
                meta=meta,
                metrics_by_split={"val": metrics_val},
                baseline_val=None,
                delta=None,
                index_path=index,
            )
            report = (out / "report.md").read_text()
            self.assertIn("unknown wave w1_bogus", report)
            # "smoke" must not silently appear as a fabricated focus row
            focus_section = report.split("### Focus classes")[1].split("### Base classes")[0]
            self.assertNotIn("smoke", focus_section)


if __name__ == "__main__":
    unittest.main()
