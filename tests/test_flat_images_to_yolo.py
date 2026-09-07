"""flat_images_to_yolo：纯类商品图 → YOLO detect 布局 + 前景框。"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def _load_mod():
    path = ROOT / "datasets/scripts/flat_images_to_yolo.py"
    spec = importlib.util.spec_from_file_location("flat_images_to_yolo", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestFlatImagesToYolo(unittest.TestCase):
    def test_foreground_bbox_on_white_background(self):
        mod = _load_mod()
        im = Image.new("RGB", (100, 80), (255, 255, 255))
        for x in range(20, 70):
            for y in range(10, 50):
                im.putpixel((x, y), (10, 20, 200))
        xc, yc, nw, nh = mod.foreground_yolo_box(im)
        self.assertAlmostEqual(xc, 0.45, delta=0.05)
        self.assertAlmostEqual(yc, 0.375, delta=0.05)
        self.assertGreater(nw, 0.4)
        self.assertLess(nw, 0.7)
        self.assertGreater(nh, 0.4)
        self.assertLess(nh, 0.7)

    def test_convert_writes_splits_and_ebike_labels(self):
        mod = _load_mod()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = td / "e-bike"
            src.mkdir()
            for i in range(10):
                im = Image.new("RGB", (80, 60), (255, 255, 255))
                for x in range(10, 50):
                    for y in range(8, 40):
                        im.putpixel((x, y), (30, 30, 30))
                im.save(src / f"img{i:02d}.jpg")
            out = td / "e-bike_yolo"
            meta = mod.convert_flat_images(
                src,
                out,
                class_name="ebike",
                split_ratios=(0.7, 0.2, 0.1),
                seed=0,
                copy_images=True,
            )
            cfg = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
            names = cfg["names"]
            if isinstance(names, dict):
                names = [names[k] for k in sorted(int(x) for x in names)]
            self.assertEqual(list(names), ["ebike"])
            n = sum(meta["images_written"].values())
            self.assertEqual(n, 10)
            lbls = list((out / "labels").rglob("*.txt"))
            self.assertEqual(len(lbls), 10)
            line = lbls[0].read_text(encoding="utf-8").strip().split()
            self.assertEqual(line[0], "0")
            self.assertEqual(len(line), 5)


if __name__ == "__main__":
    unittest.main()
