# patrol_perception/tests/test_add_negatives_to_joint.py
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_mod(name: str = "add_negatives_to_joint"):
    path = ROOT / f"datasets/scripts/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestDiscoverSample(unittest.TestCase):
    def test_is_empty_label(self):
        mod = _load_mod()
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            missing = d / "no.txt"
            empty = d / "e.txt"
            blank = d / "b.txt"
            pos = d / "p.txt"
            empty.write_text("", encoding="utf-8")
            blank.write_text("  \n", encoding="utf-8")
            pos.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
            self.assertTrue(mod.is_empty_label(missing))
            self.assertTrue(mod.is_empty_label(empty))
            self.assertTrue(mod.is_empty_label(blank))
            self.assertFalse(mod.is_empty_label(pos))

    def test_collect_empty_from_yolo_and_images(self):
        mod = _load_mod()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "yolo"
            for split in ("train", "val"):
                (root / "images" / split).mkdir(parents=True)
                (root / "labels" / split).mkdir(parents=True)
            img_neg = root / "images" / "train" / "a.jpg"
            img_pos = root / "images" / "train" / "b.jpg"
            img_neg.write_bytes(b"x")
            img_pos.write_bytes(b"x")
            (root / "labels" / "train" / "a.txt").write_text("", encoding="utf-8")
            (root / "labels" / "train" / "b.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )
            empties = mod.collect_empty_from_yolo(root)
            self.assertEqual(empties, [img_neg.resolve()])

            flat = Path(tmp) / "flat"
            (flat / "sub").mkdir(parents=True)
            p1 = flat / "c.jpg"
            p2 = flat / "sub" / "d.png"
            p1.write_bytes(b"x")
            p2.write_bytes(b"x")
            got = sorted(mod.collect_images_recursive(flat))
            self.assertEqual(got, sorted([p1.resolve(), p2.resolve()]))

    def test_collect_empty_from_yolo_without_labels_directory(self):
        mod = _load_mod()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "yolo"
            image_dir = root / "images" / "train"
            image_dir.mkdir(parents=True)
            images = [image_dir / "a.jpg", image_dir / "b.jpg"]
            for image in images:
                image.write_bytes(b"x")

            self.assertEqual(
                sorted(mod.collect_empty_from_yolo(root)),
                sorted(image.resolve() for image in images),
            )

    def test_count_positives_and_target(self):
        mod = _load_mod()
        with tempfile.TemporaryDirectory() as tmp:
            joint = Path(tmp) / "joint"
            for split, n_pos, n_neg in (("train", 3, 1), ("val", 1, 0), ("test", 1, 0)):
                (joint / "images" / split).mkdir(parents=True)
                (joint / "labels" / split).mkdir(parents=True)
                for i in range(n_pos):
                    (joint / "images" / split / f"p{i}.jpg").write_bytes(b"x")
                    (joint / "labels" / split / f"p{i}.txt").write_text(
                        "0 0.5 0.5 0.1 0.1\n", encoding="utf-8"
                    )
                for i in range(n_neg):
                    (joint / "images" / split / f"n{i}.jpg").write_bytes(b"x")
                    (joint / "labels" / split / f"n{i}.txt").write_text("", encoding="utf-8")
            self.assertEqual(mod.count_positive_images(joint), 5)
            self.assertEqual(
                mod.compute_target_count(candidates=100, positives=5, ratio=0.3, max_n=10),
                2,
            )  # ceil(1.5)=2, min(100,2,10)=2
            self.assertEqual(
                mod.compute_target_count(candidates=100, positives=5, ratio=None, max_n=None),
                100,
            )
            self.assertEqual(
                mod.compute_target_count(candidates=3, positives=5, ratio=1.0, max_n=10),
                3,
            )

    def test_sample_paths_deterministic(self):
        mod = _load_mod()
        paths = [Path(f"/tmp/{i}.jpg") for i in range(10)]
        a = mod.sample_paths(paths, 4, seed=0)
        b = mod.sample_paths(paths, 4, seed=0)
        c = mod.sample_paths(paths, 4, seed=1)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 4)
        self.assertNotEqual(a, c)


class TestStaging(unittest.TestCase):
    def test_write_neg_staging_empty_labels(self):
        mod = _load_mod()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            src.mkdir()
            imgs = []
            for i in range(5):
                p = src / f"im{i}.jpg"
                p.write_bytes(b"x")
                imgs.append(p.resolve())
            staging = Path(tmp) / "staging"
            by_split = mod.split_pool(imgs, ratios=[0.6, 0.2, 0.2], seed=0)
            meta = mod.write_neg_staging(
                staging=staging, by_split=by_split, names=["person", "car"]
            )
            self.assertEqual(sum(len(v) for v in by_split.values()), 5)
            total = 0
            for split, paths in by_split.items():
                for _ in paths:
                    total += 1
                idir = staging / "images" / split
                ldir = staging / "labels" / split
                if paths:
                    self.assertTrue(idir.is_dir())
                    for img in idir.iterdir():
                        lbl = ldir / f"{img.stem}.txt"
                        self.assertTrue(lbl.is_file())
                        self.assertEqual(lbl.read_text(encoding="utf-8"), "")
            self.assertEqual(total, 5)
            self.assertTrue((staging / "data.yaml").is_file())


class TestCliEndToEnd(unittest.TestCase):
    def test_staging_equal_from_images_preserves_source(self):
        mod = _load_mod()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "source"
            source.mkdir()
            source_image = source / "negative.jpg"
            source_image.write_bytes(b"negative")

            rc = mod.main(
                [
                    "--out",
                    str(base / "joint"),
                    "--from-images",
                    str(source),
                    "--staging",
                    str(source),
                ]
            )

            self.assertEqual(rc, 2)
            self.assertEqual(source_image.read_bytes(), b"negative")

    def test_staging_inside_out_is_rejected(self):
        mod = _load_mod()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "source"
            source.mkdir()
            source_image = source / "negative.jpg"
            source_image.write_bytes(b"negative")
            out = base / "joint"

            rc = mod.main(
                [
                    "--out",
                    str(out),
                    "--from-images",
                    str(source),
                    "--staging",
                    str(out / "staging"),
                ]
            )

            self.assertEqual(rc, 2)
            self.assertEqual(source_image.read_bytes(), b"negative")

    def test_staging_equal_out_preserves_joint_positives(self):
        mod = _load_mod()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            joint = base / "joint"
            positive_image = joint / "images" / "train" / "positive.jpg"
            positive_label = joint / "labels" / "train" / "positive.txt"
            positive_image.parent.mkdir(parents=True)
            positive_label.parent.mkdir(parents=True)
            positive_image.write_bytes(b"positive")
            positive_label.write_text(
                "0 0.5 0.5 0.1 0.1\n", encoding="utf-8"
            )
            (joint / "data.yaml").write_text(
                "names: [person]\n", encoding="utf-8"
            )

            source = base / "source"
            source.mkdir()
            (source / "negative.jpg").write_bytes(b"negative")

            rc = mod.main(
                [
                    "--out",
                    str(joint),
                    "--from-images",
                    str(source),
                    "--staging",
                    str(joint),
                ]
            )

            self.assertEqual(rc, 2)
            self.assertEqual(positive_image.read_bytes(), b"positive")
            self.assertEqual(
                positive_label.read_text(encoding="utf-8"),
                "0 0.5 0.5 0.1 0.1\n",
            )

    def test_dry_run_and_append_empty(self):
        mod = _load_mod()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            joint = base / "detect_w1_fire"
            for split in ("train", "val", "test"):
                (joint / "images" / split).mkdir(parents=True)
                (joint / "labels" / split).mkdir(parents=True)
            for i in range(2):
                (joint / "images" / "train" / f"p{i}.jpg").write_bytes(b"x")
                (joint / "labels" / "train" / f"p{i}.txt").write_text(
                    "0 0.5 0.5 0.1 0.1\n", encoding="utf-8"
                )
            (joint / "data.yaml").write_text(
                "names: [person, car]\n", encoding="utf-8"
            )
            (joint / "merge_meta.json").write_text(
                '{"sets": [{"prefix": "seed", "keep": ["person"]}]}\n',
                encoding="utf-8",
            )

            yolo = base / "src_yolo"
            (yolo / "images" / "train").mkdir(parents=True)
            (yolo / "labels" / "train").mkdir(parents=True)
            for i in range(4):
                (yolo / "images" / "train" / f"n{i}.jpg").write_bytes(b"x")
                (yolo / "labels" / "train" / f"n{i}.txt").write_text(
                    "", encoding="utf-8"
                )
            staging = base / "staging"

            rc = mod.main(
                [
                    "--out",
                    str(joint),
                    "--from-yolo",
                    str(yolo),
                    "--staging",
                    str(staging),
                    "--ratio",
                    "1.0",
                    "--max",
                    "3",
                    "--seed",
                    "0",
                    "--dry-run",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertFalse(staging.exists())

            rc = mod.main(
                [
                    "--out",
                    str(joint),
                    "--from-yolo",
                    str(yolo),
                    "--staging",
                    str(staging),
                    "--ratio",
                    "1.0",
                    "--max",
                    "3",
                    "--seed",
                    "0",
                    "--prefix",
                    "bg",
                ]
            )
            self.assertEqual(rc, 0)

            empties = 0
            for split in ("train", "val", "test"):
                idir = joint / "images" / split
                if not idir.is_dir():
                    continue
                for img in idir.iterdir():
                    if img.name.startswith("bg_"):
                        lbl = joint / "labels" / split / f"{img.stem}.txt"
                        self.assertTrue(mod.is_empty_label(lbl))
                        empties += 1
            # T=min(C=4, ceil(P=2 * R=1), max=3) = 2.
            self.assertEqual(empties, 2)

            rc = mod.main(
                [
                    "--out",
                    str(joint),
                    "--from-yolo",
                    str(yolo),
                    "--staging",
                    str(staging),
                    "--max",
                    "1",
                    "--prefix",
                    "bg",
                ]
            )
            self.assertEqual(rc, 2)

            rc = mod.main(
                [
                    "--out",
                    str(joint),
                    "--prefix",
                    "bg",
                    "--remove",
                ]
            )
            self.assertEqual(rc, 0)
            for split in ("train", "val", "test"):
                idir = joint / "images" / split
                if not idir.is_dir():
                    continue
                self.assertFalse(any(p.name.startswith("bg_") for p in idir.iterdir()))
            self.assertTrue(
                (joint / "images" / "train" / "p0.jpg").is_file()
            )
            self.assertFalse(mod.prefix_present(joint, "bg"))
