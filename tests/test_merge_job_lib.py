"""merge_job_lib：job 登记表加载与 lookup。"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_mod(name: str = "merge_job_lib"):
    path = ROOT / f"datasets/scripts/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_out_state(
    out_root: Path, *, names: list[str] | dict[int, str], sets: list[dict]
) -> None:
    out_root.mkdir(parents=True)
    (out_root / "data.yaml").write_text(
        yaml.safe_dump({"names": names}, sort_keys=False), encoding="utf-8"
    )
    (out_root / "merge_meta.json").write_text(
        json.dumps({"sets": sets}), encoding="utf-8"
    )


class TestMergeJobLib(unittest.TestCase):
    def test_load_jobs_registry_coco_pet(self):
        mod = _load_mod("merge_job_lib")
        reg = mod.load_jobs_registry(ROOT / "config/merge_jobs.yaml")
        job = reg["coco_pet"]
        self.assertEqual(job.prefix, "coco_pet")
        self.assertEqual(job.candidates, ["pet"])
        self.assertTrue(job.map_rel)
        self.assertTrue(job.extra_remap_out_rel)
        self.assertIn("coco_train2017", job.src_rel)
        self.assertTrue(job.extra_roots_rel)  # coco_val2017

    def test_get_job_returns_spec(self):
        mod = _load_mod("merge_job_lib")
        reg = mod.load_jobs_registry(ROOT / "config/merge_jobs.yaml")
        job = mod.get_job(reg, "bicycle2")
        self.assertEqual(job.name, "bicycle2")
        self.assertEqual(job.prefix, "bike2")
        self.assertEqual(job.candidates, ["bicycle"])

    def test_get_job_missing_raises(self):
        mod = _load_mod("merge_job_lib")
        reg = mod.load_jobs_registry(ROOT / "config/merge_jobs.yaml")
        with self.assertRaises(KeyError):
            mod.get_job(reg, "no_such_job")

    def test_registry_has_all_jobs(self):
        mod = _load_mod("merge_job_lib")
        reg = mod.load_jobs_registry(ROOT / "config/merge_jobs.yaml")
        expected = {
            "fire",
            "coco",
            "coco_pet",
            "taco",
            "fishing",
            "cig",
            "ebike",
            "fishing2",
            "bicycle2",
            "cigarette2",
            "ebike2",
            "trash2",
            "pet2",
        }
        self.assertEqual(set(reg.keys()), expected)
        coco = reg["coco"]
        self.assertEqual(coco.candidates, ["person", "bicycle", "car", "truck"])
        self.assertEqual(coco.prefix, "coco")
        self.assertTrue(coco.extra_roots_rel)
        self.assertEqual(reg["fire"].candidates, ["flame", "smoke"])
        trash = reg["trash2"]
        self.assertEqual(trash.prefix, "trash2")
        self.assertEqual(trash.split_ratios, [0.7, 0.2, 0.1])
        self.assertEqual(
            trash.candidates,
            [
                "litter_bottle",
                "litter_plastic_bag",
                "litter_paper",
                "litter_cardboard",
            ],
        )

    def test_prefix_keep_absent_and_present(self):
        mod = _load_mod("merge_job_lib")
        meta = {"sets": [{"prefix": "coco_pet", "keep": ["cat", "dog"]}]}
        self.assertIsNone(mod.prefix_keep(meta, "taco"))
        self.assertEqual(mod.prefix_keep(meta, "coco_pet"), ["cat", "dog"])

    def test_prefix_keep_allows_explicit_empty_list(self):
        mod = _load_mod("merge_job_lib")
        meta = {"sets": [{"prefix": "coco_pet", "keep": []}]}
        self.assertEqual(mod.prefix_keep(meta, "coco_pet"), [])

    def test_prefix_keep_rejects_missing_or_null_keep(self):
        mod = _load_mod("merge_job_lib")
        for merge_set in (
            {"prefix": "coco_pet"},
            {"prefix": "coco_pet", "keep": None},
        ):
            with self.subTest(merge_set=merge_set):
                with self.assertRaises(ValueError):
                    mod.prefix_keep({"sets": [merge_set]}, "coco_pet")

    def test_extend_names_appends_in_candidate_order(self):
        mod = _load_mod("merge_job_lib")
        base = ["person", "cat"]
        out = mod.extend_names(base, keep=["dog", "cat"], candidates=["cat", "dog"])
        self.assertEqual(out, ["person", "cat", "dog"])

    def test_resolve_keep_yes_and_subset(self):
        mod = _load_mod("merge_job_lib")
        self.assertEqual(
            mod.resolve_keep(
                candidates=["cat", "dog"], keep_arg=None, yes=True, input_fn=None
            ),
            ["cat", "dog"],
        )
        self.assertEqual(
            mod.resolve_keep(
                candidates=["cat", "dog"], keep_arg=["dog"], yes=False, input_fn=None
            ),
            ["dog"],
        )

    def test_resolve_drop_interactive(self):
        mod = _load_mod("merge_job_lib")
        got = mod.resolve_drop(
            current=["cat", "dog"], drop_arg=None, yes=False, input_fn=lambda _: "2"
        )
        self.assertEqual(got, ["dog"])

    def test_resolve_drop_single_class_requires_interactive_choice(self):
        mod = _load_mod("merge_job_lib")
        calls = []

        def choose_first(prompt):
            calls.append(prompt)
            return "1"

        got = mod.resolve_drop(
            current=["fishing_rod"],
            drop_arg=None,
            yes=False,
            input_fn=choose_first,
        )

        self.assertEqual(got, ["fishing_rod"])
        self.assertEqual(len(calls), 1)

    def test_plan_append_skips_when_present(self):
        mod = _load_mod("merge_job_lib")
        job = mod.load_jobs_registry(ROOT / "config/merge_jobs.yaml")["coco_pet"]
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp) / "out"
            _write_out_state(
                out_root,
                names=["person", "cat", "dog"],
                sets=[{"prefix": "coco_pet", "keep": ["cat", "dog"]}],
            )

            plan = mod.plan_append(
                out_root=out_root,
                data_root=Path(tmp) / "data",
                job=job,
                keep_arg=["cat"],
                yes=False,
                input_fn=None,
            )

        self.assertIsInstance(plan, mod.SkipPlan)
        self.assertIn("coco_pet", plan.reason)

    def test_plan_append_builds_keep_subset(self):
        mod = _load_mod("merge_job_lib")
        job = mod.load_jobs_registry(ROOT / "config/merge_jobs.yaml")["coco_pet"]
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "data"
            out_root = Path(tmp) / "out"
            _write_out_state(out_root, names=["person"], sets=[])

            plan = mod.plan_append(
                out_root=out_root,
                data_root=data_root,
                job=job,
                keep_arg=["pet"],
                yes=False,
                input_fn=None,
            )

        self.assertIsInstance(plan, mod.AppendPlan)
        self.assertEqual(plan.keep, ["pet"])
        self.assertEqual(plan.names, ["person", "pet"])
        self.assertTrue(plan.need_remap)
        self.assertEqual(plan.src, data_root / job.src_rel)
        self.assertEqual(plan.remap_out, data_root / job.remap_out_rel)
        self.assertIsNotNone(plan.map_path)
        self.assertEqual(len(plan.extra_remaps), 1)
        merge_set = plan.merge_yaml["sets"][0]
        self.assertEqual(merge_set["root"], str(data_root / job.remap_out_rel))
        self.assertEqual(
            merge_set["extra_roots"],
            [str(data_root / rel) for rel in job.extra_remap_out_rel],
        )
        self.assertEqual(merge_set["keep"], ["pet"])
        self.assertEqual(merge_set["split_ratios"], job.split_ratios)
        self.assertEqual(merge_set["split_seed"], job.split_seed)

    def test_plan_remove_partial_reappend(self):
        mod = _load_mod("merge_job_lib")
        job = mod.load_jobs_registry(ROOT / "config/merge_jobs.yaml")["taco"]
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "data"
            out_root = Path(tmp) / "out"
            _write_out_state(
                out_root,
                names={0: "person", 1: "litter_bottle", 2: "litter_plastic_bag"},
                sets=[
                    {
                        "prefix": "taco",
                        "keep": ["litter_bottle", "litter_plastic_bag"],
                    }
                ],
            )

            plan = mod.plan_remove(
                out_root=out_root,
                data_root=data_root,
                job=job,
                drop_arg=["litter_plastic_bag"],
                yes=False,
                input_fn=None,
            )

        self.assertIsInstance(plan, mod.RemovePlan)
        self.assertEqual(plan.prefix, "taco")
        self.assertEqual(plan.drop, ["litter_plastic_bag"])
        self.assertEqual(plan.remain, ["litter_bottle"])
        self.assertIsInstance(plan.reappend, mod.AppendPlan)
        self.assertEqual(plan.reappend.keep, ["litter_bottle"])
        self.assertEqual(
            plan.reappend.names,
            ["person", "litter_bottle", "litter_plastic_bag"],
        )

    def test_plan_remove_yes_no_reappend(self):
        mod = _load_mod("merge_job_lib")
        job = mod.load_jobs_registry(ROOT / "config/merge_jobs.yaml")["coco_pet"]
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp) / "out"
            _write_out_state(
                out_root,
                names=["person", "pet"],
                sets=[{"prefix": "coco_pet", "keep": ["pet"]}],
            )

            plan = mod.plan_remove(
                out_root=out_root,
                data_root=Path(tmp) / "data",
                job=job,
                drop_arg=None,
                yes=True,
                input_fn=None,
            )

        self.assertIsInstance(plan, mod.RemovePlan)
        self.assertEqual(plan.drop, ["pet"])
        self.assertEqual(plan.remain, [])
        self.assertIsNone(plan.reappend)
