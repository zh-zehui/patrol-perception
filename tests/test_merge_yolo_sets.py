"""Task 3: merge_yolo_sets — 人造两套 YOLO 标签合并后全局 id 正确。"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_merge_mod():
    path = ROOT / "datasets/scripts/merge_yolo_sets.py"
    spec = importlib.util.spec_from_file_location("merge_yolo_sets", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_yolo_set(
    root: Path,
    *,
    names: dict[int, str],
    train_boxes: dict[str, str],
    val_boxes: dict[str, str] | None = None,
) -> Path:
    """train_boxes: stem -> label file text."""
    for split, boxes in (("train", train_boxes), ("val", val_boxes or {})):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for stem, text in boxes.items():
            (root / "images" / split / f"{stem}.jpg").write_bytes(b"img")
            (root / "labels" / split / f"{stem}.txt").write_text(text, encoding="utf-8")
    cfg = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val" if val_boxes else "images/train",
        "names": names,
    }
    (root / "data.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )
    return root


class TestMergeYoloSets(unittest.TestCase):
    def test_remap_label_file_id_map(self):
        mod = _load_merge_mod()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = td / "a.txt"
            dst = td / "b.txt"
            src.write_text("0 0.1 0.2 0.3 0.4\n2 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.2 0.2\n", encoding="utf-8")
            # 只保留 0→3, 2→7；丢弃 1
            kept, dropped = mod.remap_label_file(src, dst, {0: 3, 2: 7})
            self.assertEqual(kept, 2)
            self.assertEqual(dropped, 1)
            lines = dst.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(lines[0].split()[0], "3")
            self.assertEqual(lines[1].split()[0], "7")

    def test_merge_two_sets_global_ids(self):
        mod = _load_merge_mod()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            fire = _make_yolo_set(
                td / "fire",
                names={0: "flame", 1: "smoke"},
                train_boxes={"f1": "0 0.1 0.2 0.3 0.4\n1 0.5 0.5 0.1 0.1\n"},
                val_boxes={"fv": "1 0.4 0.4 0.2 0.2\n"},
            )
            coco = _make_yolo_set(
                td / "coco",
                names={0: "person", 1: "bicycle", 2: "car", 7: "truck"},
                train_boxes={"c1": "0 0.2 0.2 0.1 0.1\n2 0.3 0.3 0.1 0.1\n"},
                val_boxes={"cv": "7 0.6 0.6 0.1 0.1\n"},
            )
            out = td / "merged"
            global_names = ["person", "bicycle", "car", "truck", "smoke", "flame"]
            meta = mod.merge_sets(
                [
                    {"root": fire, "prefix": "fire", "keep": ["flame", "smoke"]},
                    {"root": coco, "prefix": "coco", "keep": ["person", "bicycle", "car", "truck"]},
                ],
                out,
                global_names,
                copy_images=True,
            )
            self.assertTrue((out / "data.yaml").is_file())
            cfg = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
            names = cfg["names"]
            if isinstance(names, dict):
                names = [names[i] for i in sorted(int(k) for k in names)]
            self.assertEqual(list(names), global_names)

            # flame→5, smoke→4
            fire_lbl = (out / "labels/train/fire_f1.txt").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(fire_lbl[0].split()[0], "5")  # flame
            self.assertEqual(fire_lbl[1].split()[0], "4")  # smoke

            # person→0, car→2
            coco_lbl = (out / "labels/train/coco_c1.txt").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(coco_lbl[0].split()[0], "0")
            self.assertEqual(coco_lbl[1].split()[0], "2")

            self.assertTrue((out / "images/train/fire_f1.jpg").is_file())
            self.assertTrue((out / "images/val/coco_cv.jpg").is_file())
            self.assertGreaterEqual(meta["images_written"]["train"], 2)
            self.assertGreaterEqual(meta["images_written"]["val"], 2)

    def test_coco_val2017_split_no_leakage(self):
        """仅有 val2017 时应按比例拆成 train/val，且文件不重叠。"""
        mod = _load_merge_mod()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            root = td / "coco"
            (root / "images" / "val2017").mkdir(parents=True)
            (root / "labels" / "val2017").mkdir(parents=True)
            for i in range(20):
                (root / "images" / "val2017" / f"{i:02d}.jpg").write_bytes(b"x")
                (root / "labels" / "val2017" / f"{i:02d}.txt").write_text(
                    "0 0.1 0.1 0.1 0.1\n", encoding="utf-8"
                )
            (root / "data.yaml").write_text(
                yaml.safe_dump(
                    {
                        "path": str(root),
                        "train": "images/val2017",
                        "val": "images/val2017",
                        "names": {0: "person"},
                    }
                ),
                encoding="utf-8",
            )
            out = td / "out"
            meta = mod.merge_sets(
                [
                    {
                        "root": root,
                        "prefix": "c",
                        "keep": ["person"],
                        "val_fraction": 0.2,
                        "split_seed": 0,
                    }
                ],
                out,
                ["person"],
                copy_images=True,
            )
            train = {p.name for p in (out / "images/train").iterdir()}
            val = {p.name for p in (out / "images/val").iterdir()}
            self.assertGreaterEqual(len(train), 14)
            self.assertGreaterEqual(len(val), 3)
            self.assertEqual(train & val, set())
            self.assertEqual(len(train) + len(val), 20)
            self.assertEqual(meta["images_written"]["train"] + meta["images_written"]["val"], 20)

    def test_assign_split_val_keeps_entire_pool(self):
        """单池 + assign_split=val：全部进 val，不按 val_fraction 拆。"""
        mod = _load_merge_mod()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            root = td / "coco"
            (root / "images" / "val2017").mkdir(parents=True)
            (root / "labels" / "val2017").mkdir(parents=True)
            for i in range(10):
                (root / "images" / "val2017" / f"{i:02d}.jpg").write_bytes(b"x")
                (root / "labels" / "val2017" / f"{i:02d}.txt").write_text(
                    "0 0.1 0.1 0.1 0.1\n", encoding="utf-8"
                )
            (root / "data.yaml").write_text(
                yaml.safe_dump(
                    {
                        "path": str(root),
                        "val": "images/val2017",
                        "names": {0: "person"},
                    }
                ),
                encoding="utf-8",
            )
            out = td / "out"
            mod.merge_sets(
                [
                    {
                        "root": root,
                        "prefix": "c",
                        "keep": ["person"],
                        "assign_split": "val",
                    }
                ],
                out,
                ["person"],
                copy_images=True,
            )
            self.assertTrue((out / "images/val").is_dir())
            self.assertEqual(len(list((out / "images/val").iterdir())), 10)
            self.assertFalse((out / "images/train").exists() and any((out / "images/train").iterdir()))

    def test_split_ratios_721_pools_train_and_val_roots(self):
        """COCO train+val 两根目录合成一池后按 7:2:1 无重叠划分。"""
        mod = _load_merge_mod()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            train_root = td / "coco_train"
            val_root = td / "coco_val"
            for root, dirname, n0, n1 in (
                (train_root, "train2017", 0, 70),
                (val_root, "val2017", 70, 100),
            ):
                (root / "images" / dirname).mkdir(parents=True)
                (root / "labels" / dirname).mkdir(parents=True)
                for i in range(n0, n1):
                    (root / "images" / dirname / f"{i:03d}.jpg").write_bytes(b"x")
                    (root / "labels" / dirname / f"{i:03d}.txt").write_text(
                        "0 0.1 0.1 0.1 0.1\n", encoding="utf-8"
                    )
                (root / "data.yaml").write_text(
                    yaml.safe_dump(
                        {
                            "path": str(root),
                            "train" if dirname.startswith("train") else "val": f"images/{dirname}",
                            "names": {0: "person", 1: "bicycle", 2: "car", 7: "truck"},
                        }
                    ),
                    encoding="utf-8",
                )
            out = td / "out"
            meta = mod.merge_sets(
                [
                    {
                        "root": train_root,
                        "extra_roots": [str(val_root)],
                        "prefix": "coco",
                        "keep": ["person", "bicycle", "car", "truck"],
                        "split_ratios": [0.7, 0.2, 0.1],
                        "split_seed": 0,
                    }
                ],
                out,
                ["person", "bicycle", "car", "truck"],
                copy_images=True,
            )
            train = {p.name for p in (out / "images/train").iterdir()}
            val = {p.name for p in (out / "images/val").iterdir()}
            test = {p.name for p in (out / "images/test").iterdir()}
            self.assertEqual(len(train) + len(val) + len(test), 100)
            self.assertEqual(train & val, set())
            self.assertEqual(train & test, set())
            self.assertEqual(val & test, set())
            self.assertEqual(len(train), 70)
            self.assertEqual(len(val), 20)
            self.assertEqual(len(test), 10)
            self.assertEqual(meta["images_written"]["train"], 70)
            self.assertEqual(meta["images_written"]["test"], 10)

    def test_fire_keeps_native_train_val_test(self):
        """烟火已有三分法时默认原样进入合并集，不强制 7:2:1。"""
        mod = _load_merge_mod()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            fire = _make_yolo_set(
                td / "fire",
                names={0: "flame", 1: "smoke"},
                train_boxes={"a": "0 0.1 0.1 0.1 0.1\n"},
                val_boxes={"b": "1 0.2 0.2 0.1 0.1\n"},
            )
            (fire / "images" / "test").mkdir(parents=True)
            (fire / "labels" / "test").mkdir(parents=True)
            (fire / "images" / "test" / "c.jpg").write_bytes(b"x")
            (fire / "labels" / "test" / "c.txt").write_text(
                "1 0.3 0.3 0.1 0.1\n", encoding="utf-8"
            )
            cfg = yaml.safe_load((fire / "data.yaml").read_text(encoding="utf-8"))
            cfg["test"] = "images/test"
            (fire / "data.yaml").write_text(
                yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
            )
            out = td / "out"
            meta = mod.merge_sets(
                [{"root": fire, "prefix": "fire", "keep": ["flame", "smoke"]}],
                out,
                ["smoke", "flame"],
                copy_images=True,
            )
            self.assertTrue((out / "images/train/fire_a.jpg").is_file())
            self.assertTrue((out / "images/val/fire_b.jpg").is_file())
            self.assertTrue((out / "images/test/fire_c.jpg").is_file())
            self.assertEqual(meta["images_written"]["train"], 1)
            self.assertEqual(meta["images_written"]["val"], 1)
            self.assertEqual(meta["images_written"]["test"], 1)

    def test_merge_drops_unkept_classes(self):
        mod = _load_merge_mod()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = _make_yolo_set(
                td / "src",
                names={0: "person", 1: "dog", 2: "car"},
                train_boxes={"a": "0 0.1 0.1 0.1 0.1\n1 0.2 0.2 0.1 0.1\n2 0.3 0.3 0.1 0.1\n"},
            )
            out = td / "out"
            mod.merge_sets(
                [{"root": src, "prefix": "s", "keep": ["person", "car"]}],
                out,
                ["person", "car"],
                copy_images=True,
                keep_empty=False,
            )
            text = (out / "labels/train/s_a.txt").read_text(encoding="utf-8").strip().splitlines()
            ids = [int(float(l.split()[0])) for l in text]
            self.assertEqual(ids, [0, 1])  # person, car in global order

    def test_append_extends_names_keeps_old_files(self):
        """append：旧图/旧标签 id 不动，新类挂在 names 末尾。"""
        mod = _load_merge_mod()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            base = _make_yolo_set(
                td / "base",
                names={0: "person"},
                train_boxes={"p1": "0 0.1 0.1 0.1 0.1\n"},
            )
            out = td / "out"
            mod.merge_sets(
                [{"root": base, "prefix": "base", "keep": ["person"]}],
                out,
                ["person"],
                copy_images=True,
            )
            old_lbl = (out / "labels/train/base_p1.txt").read_text(encoding="utf-8")

            ebike = _make_yolo_set(
                td / "ebike",
                names={0: "person", 1: "ebike"},
                train_boxes={"e1": "1 0.2 0.2 0.1 0.1\n"},
            )
            meta = mod.merge_sets(
                [{"root": ebike, "prefix": "ebike", "keep": ["person", "ebike"]}],
                out,
                ["person", "ebike"],
                copy_images=True,
                mode="append",
            )
            self.assertEqual(
                (out / "labels/train/base_p1.txt").read_text(encoding="utf-8"),
                old_lbl,
            )
            self.assertTrue((out / "images/train/base_p1.jpg").is_file())
            self.assertTrue((out / "images/train/ebike_e1.jpg").is_file())
            e_lbl = (out / "labels/train/ebike_e1.txt").read_text(encoding="utf-8").strip()
            self.assertEqual(e_lbl.split()[0], "1")  # ebike → id 1
            cfg = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
            names = cfg["names"]
            if isinstance(names, dict):
                names = [names[i] for i in sorted(int(k) for k in names)]
            self.assertEqual(list(names), ["person", "ebike"])
            self.assertEqual(meta.get("mode"), "append")

    def test_append_rejects_non_prefix_names(self):
        mod = _load_merge_mod()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            base = _make_yolo_set(
                td / "base",
                names={0: "person"},
                train_boxes={"p1": "0 0.1 0.1 0.1 0.1\n"},
            )
            out = td / "out"
            mod.merge_sets(
                [{"root": base, "prefix": "base", "keep": ["person"]}],
                out,
                ["person"],
                copy_images=True,
            )
            extra = _make_yolo_set(
                td / "extra",
                names={0: "dog"},
                train_boxes={"d1": "0 0.1 0.1 0.1 0.1\n"},
            )
            with self.assertRaises(ValueError):
                mod.merge_sets(
                    [{"root": extra, "prefix": "x", "keep": ["dog"]}],
                    out,
                    ["dog", "person"],  # 插到前面 / 非前缀
                    copy_images=True,
                    mode="append",
                )

    def test_append_collision_raises(self):
        mod = _load_merge_mod()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            base = _make_yolo_set(
                td / "base",
                names={0: "person"},
                train_boxes={"a": "0 0.1 0.1 0.1 0.1\n"},
            )
            out = td / "out"
            mod.merge_sets(
                [{"root": base, "prefix": "p", "keep": ["person"]}],
                out,
                ["person"],
                copy_images=True,
            )
            again = _make_yolo_set(
                td / "again",
                names={0: "person"},
                train_boxes={"a": "0 0.2 0.2 0.1 0.1\n"},
            )
            with self.assertRaises(FileExistsError):
                mod.merge_sets(
                    [{"root": again, "prefix": "p", "keep": ["person"]}],
                    out,
                    ["person"],
                    copy_images=True,
                    mode="append",
                )
            self.assertTrue((out / "images/train/p_a.jpg").is_file())

    def test_full_mode_still_clears_out(self):
        """默认 full：再次合并会清空旧 out。"""
        mod = _load_merge_mod()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            a = _make_yolo_set(
                td / "a",
                names={0: "person"},
                train_boxes={"old": "0 0.1 0.1 0.1 0.1\n"},
            )
            out = td / "out"
            mod.merge_sets(
                [{"root": a, "prefix": "a", "keep": ["person"]}],
                out,
                ["person"],
                copy_images=True,
            )
            b = _make_yolo_set(
                td / "b",
                names={0: "person"},
                train_boxes={"new": "0 0.1 0.1 0.1 0.1\n"},
            )
            mod.merge_sets(
                [{"root": b, "prefix": "b", "keep": ["person"]}],
                out,
                ["person"],
                copy_images=True,
                mode="full",
            )
            self.assertFalse((out / "images/train/a_old.jpg").exists())
            self.assertTrue((out / "images/train/b_new.jpg").is_file())

    def test_remove_drops_prefix_keeps_others_and_names(self):
        """remove：只删指定 prefix 文件，其它图/标签与 names 不动。"""
        mod = _load_merge_mod()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            base = _make_yolo_set(
                td / "base",
                names={0: "person"},
                train_boxes={"p1": "0 0.1 0.1 0.1 0.1\n"},
            )
            extra = _make_yolo_set(
                td / "extra",
                names={0: "ebike"},
                train_boxes={"e1": "0 0.2 0.2 0.1 0.1\n"},
            )
            out = td / "out"
            mod.merge_sets(
                [{"root": base, "prefix": "base", "keep": ["person"]}],
                out,
                ["person"],
                copy_images=True,
            )
            mod.merge_sets(
                [{"root": extra, "prefix": "ebike2", "keep": ["ebike"]}],
                out,
                ["person", "ebike"],
                copy_images=True,
                mode="append",
            )
            meta = mod.merge_sets(
                [{"root": extra, "prefix": "ebike2", "keep": ["ebike"]}],
                out,
                ["person", "ebike"],
                mode="remove",
            )
            self.assertTrue((out / "images/train/base_p1.jpg").is_file())
            self.assertTrue((out / "labels/train/base_p1.txt").is_file())
            self.assertFalse((out / "images/train/ebike2_e1.jpg").exists())
            self.assertFalse((out / "labels/train/ebike2_e1.txt").exists())
            cfg = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
            names = cfg["names"]
            if isinstance(names, dict):
                names = [names[i] for i in sorted(int(k) for k in names)]
            self.assertEqual(list(names), ["person", "ebike"])
            self.assertEqual(meta.get("mode"), "remove")
            prefixes = [s.get("prefix") for s in meta.get("sets") or []]
            self.assertNotIn("ebike2", prefixes)
            self.assertIn("base", prefixes)

    def test_remove_does_not_match_shorter_prefix(self):
        """remove prefix=ebike 不得误删 ebike2_*。"""
        mod = _load_merge_mod()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            a = _make_yolo_set(
                td / "a",
                names={0: "ebike"},
                train_boxes={"x": "0 0.1 0.1 0.1 0.1\n"},
            )
            b = _make_yolo_set(
                td / "b",
                names={0: "ebike"},
                train_boxes={"y": "0 0.2 0.2 0.1 0.1\n"},
            )
            out = td / "out"
            mod.merge_sets(
                [{"root": a, "prefix": "ebike", "keep": ["ebike"]}],
                out,
                ["ebike"],
                copy_images=True,
            )
            mod.merge_sets(
                [{"root": b, "prefix": "ebike2", "keep": ["ebike"]}],
                out,
                ["ebike"],
                copy_images=True,
                mode="append",
            )
            mod.merge_sets(
                [{"prefix": "ebike"}],
                out,
                ["ebike"],
                mode="remove",
            )
            self.assertFalse((out / "images/train/ebike_x.jpg").exists())
            self.assertTrue((out / "images/train/ebike2_y.jpg").is_file())

    def test_remove_missing_prefix_is_idempotent(self):
        mod = _load_merge_mod()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            base = _make_yolo_set(
                td / "base",
                names={0: "person"},
                train_boxes={"p1": "0 0.1 0.1 0.1 0.1\n"},
            )
            out = td / "out"
            mod.merge_sets(
                [{"root": base, "prefix": "base", "keep": ["person"]}],
                out,
                ["person"],
                copy_images=True,
            )
            meta = mod.merge_sets(
                [{"prefix": "nope"}],
                out,
                ["person"],
                mode="remove",
            )
            self.assertTrue((out / "images/train/base_p1.jpg").is_file())
            self.assertEqual(sum(meta["images_removed"].values()), 0)


if __name__ == "__main__":
    unittest.main()
