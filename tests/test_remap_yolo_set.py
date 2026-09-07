import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


class TestRemapYoloSet(unittest.TestCase):
    def test_remap_keeps_cigarette_drops_face(self):
        import importlib.util

        path = ROOT / "datasets/scripts/remap_yolo_set.py"
        spec = importlib.util.spec_from_file_location("remap_yolo_set", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = td / "src"
            (src / "images/train").mkdir(parents=True)
            (src / "labels/train").mkdir(parents=True)
            (src / "images/train" / "a.jpg").write_bytes(b"x")
            (src / "labels/train" / "a.txt").write_text(
                "0 0.1 0.2 0.3 0.4\n1 0.5 0.5 0.1 0.1\n2 0.2 0.2 0.2 0.2\n",
                encoding="utf-8",
            )
            (src / "data.yaml").write_text(
                yaml.safe_dump(
                    {
                        "path": str(src),
                        "train": "images/train",
                        "names": {0: "cigarette", 1: "face", 2: "smoking"},
                    }
                ),
                encoding="utf-8",
            )
            out = td / "out"
            meta = mod.remap_yolo_set(
                src,
                out,
                ROOT / "config/class_maps/cigarette_face.yaml",
                copy_images=True,
            )
            self.assertEqual(meta["kept_boxes"], 1)
            self.assertEqual(meta["dropped_boxes"], 2)
            lbl = (out / "labels/train/a.txt").read_text().strip()
            self.assertTrue(lbl.startswith("0 "))
            names = yaml.safe_load((out / "data.yaml").read_text())["names"]
            self.assertEqual(names[0], "cigarette_or_smoke")

    def test_fire_smoke_roboflow_layout(self):
        import importlib.util

        path = ROOT / "datasets/scripts/remap_yolo_set.py"
        spec = importlib.util.spec_from_file_location("remap_yolo_set", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = td / "dataset"
            (src / "train/images").mkdir(parents=True)
            (src / "train/labels").mkdir(parents=True)
            (src / "valid/images").mkdir(parents=True)
            (src / "valid/labels").mkdir(parents=True)
            (src / "train/images/a.jpg").write_bytes(b"x")
            (src / "train/labels/a.txt").write_text(
                "0 0.1 0.2 0.3 0.4\n1 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )
            (src / "valid/images/b.jpg").write_bytes(b"y")
            (src / "valid/labels/b.txt").write_text(
                "1 0.4 0.4 0.1 0.1\n", encoding="utf-8"
            )
            # 故意写错相对路径，脚本应回退探测 train/images
            (src / "data.yaml").write_text(
                "train: ../train/images\nval: ../valid/images\nnames: ['Fire', 'Smoke']\n",
                encoding="utf-8",
            )
            out = td / "out"
            meta = mod.remap_yolo_set(
                src,
                out,
                ROOT / "config/class_maps/fire_smoke.yaml",
                copy_images=True,
            )
            self.assertEqual(meta["kept_boxes"], 3)
            self.assertEqual(meta["images_written"]["train"], 1)
            self.assertEqual(meta["images_written"]["val"], 1)
            train_lbl = (out / "labels/train/a.txt").read_text().strip().splitlines()
            self.assertEqual(train_lbl[0][0], "0")  # Fire→flame
            self.assertEqual(train_lbl[1][0], "1")  # Smoke→smoke
            names = yaml.safe_load((out / "data.yaml").read_text())["names"]
            self.assertEqual(names[0], "flame")
            self.assertEqual(names[1], "smoke")


if __name__ == "__main__":
    unittest.main()
