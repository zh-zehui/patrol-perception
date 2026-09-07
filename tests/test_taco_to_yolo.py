import json
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


class TestTacoToYolo(unittest.TestCase):
    def test_coco_bbox_to_yolo(self):
        import importlib.util

        path = ROOT / "datasets/scripts/taco_to_yolo.py"
        spec = importlib.util.spec_from_file_location("taco_to_yolo", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)

        # 100x100 图，框左上 (10,20) 宽 30 高 40 → xc=0.25 yc=0.4 w=0.3 h=0.4
        y = mod.coco_bbox_to_yolo([10, 20, 30, 40], 100, 100)
        self.assertIsNotNone(y)
        xc, yc, w, h = y
        self.assertAlmostEqual(xc, 0.25)
        self.assertAlmostEqual(yc, 0.4)
        self.assertAlmostEqual(w, 0.3)
        self.assertAlmostEqual(h, 0.4)

    def test_convert_filters_and_maps(self):
        import importlib.util

        path = ROOT / "datasets/scripts/taco_to_yolo.py"
        spec = importlib.util.spec_from_file_location("taco_to_yolo", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            taco = td / "taco"
            data = taco / "data"
            (data / "batch_1").mkdir(parents=True)
            # 最小合法 jpeg 头不够；写空文件并在 convert 里只检查 is_file
            img_path = data / "batch_1" / "000001.jpg"
            img_path.write_bytes(b"fake")

            coco = {
                "images": [
                    {
                        "id": 1,
                        "width": 100,
                        "height": 100,
                        "file_name": "batch_1/000001.jpg",
                    }
                ],
                "categories": [
                    {"id": 5, "name": "Clear plastic bottle"},
                    {"id": 59, "name": "Cigarette"},
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 5,
                        "bbox": [10, 10, 20, 20],
                        "iscrowd": 0,
                    },
                    {
                        "id": 2,
                        "image_id": 1,
                        "category_id": 59,
                        "bbox": [50, 50, 10, 10],
                        "iscrowd": 0,
                    },
                ],
            }
            (data / "annotations.json").write_text(
                json.dumps(coco), encoding="utf-8"
            )

            out = td / "out"
            meta = mod.convert_taco(
                taco,
                out,
                ROOT / "config/class_maps/taco_litter.yaml",
                val_ratio=0.0,
                seed=0,
                copy_images=True,
            )
            self.assertEqual(meta["kept_annotations"], 1)
            self.assertEqual(meta["images_train"], 1)
            lbl = (out / "labels/train/batch_1__000001.txt").read_text().strip()
            # litter_bottle id=0
            self.assertTrue(lbl.startswith("0 "))
            names = yaml.safe_load((out / "data.yaml").read_text())["names"]
            self.assertEqual(names[0], "litter_bottle")


if __name__ == "__main__":
    unittest.main()
