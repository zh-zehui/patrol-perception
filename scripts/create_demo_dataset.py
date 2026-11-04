#!/usr/bin/env python3
"""Generate geometric UI fixtures; no real images, model calls, or downloads."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def create_demo_dataset(output: Path) -> Path:
    output = output.expanduser().resolve()
    # Never overwrite an existing dataset or human annotations.
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {output}")
    images = output / "images" / "train"
    labels = output / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    width, height = 1280, 760
    for frame in range(3):
        image = Image.new("RGB", (width, height), "#e8f2fa")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=24)
        draw.rectangle((0, 370, width, height), fill="#b8d5c3")
        draw.polygon(((450, 370), (830, 370), (1180, height), (130, height)), fill="#627587")
        for y in range(410, height, 90):
            draw.rectangle((630, y, 646, y + 38), fill="#e2e8ed")
        for x, top in ((70, 145), (970, 190)):
            draw.rounded_rectangle((x, top, x + 210, 395), 10, fill="#c6d7e6")
            for wx in range(x + 22, x + 190, 55):
                for wy in range(top + 24, 355, 57):
                    draw.rectangle((wx, wy, wx + 30, wy + 34), fill="#7c9db7")
        for x in (340, 890, 1170):
            draw.rectangle((x - 8, 325, x + 8, 440), fill="#816d5f")
            draw.ellipse((x - 52, 245, x + 52, 365), fill="#4a9274")
        dx = frame * 12
        # Stylized person, car and bottle; labels are hand-authored fixtures.
        draw.ellipse((402 + dx, 390, 440 + dx, 428), fill="#e5b695")
        draw.rounded_rectangle((393 + dx, 428, 449 + dx, 508), 12, fill="#e6a642")
        draw.line((406 + dx, 503, 400 + dx, 566), fill="#263d57", width=14)
        draw.line((437 + dx, 503, 447 + dx, 566), fill="#263d57", width=14)
        draw.rounded_rectangle((745 - dx, 472, 975 - dx, 580), 24, fill="#327fb4")
        draw.polygon(((785-dx, 477), (817-dx, 429), (901-dx, 429), (940-dx, 477)), fill="#327fb4")
        draw.polygon(((804-dx, 467), (825-dx, 441), (892-dx, 441), (916-dx, 467)), fill="#b6d9ef")
        for x in (795 - dx, 926 - dx):
            draw.ellipse((x - 24, 557, x + 24, 605), fill="#23364a")
        draw.rounded_rectangle((526, 625, 546, 674), 5, fill="#66adbe")
        draw.rectangle((532, 616, 540, 628), fill="#3d7385")
        draw.rounded_rectangle((28, 24, 660, 102), 12, fill="#132d45")
        draw.text((48, 35), "SYNTHETIC DATA / UI DEMO", font=font, fill="white")
        draw.text((48, 65), "Geometric fixtures - not model predictions", font=ImageFont.load_default(size=19), fill="#b7d5e8")
        name = f"demo_{frame + 1:02d}"
        image.save(images / f"{name}.png")
        boxes = [(0, (388+dx, 386, 455+dx, 574)), (2, (741-dx, 424, 980-dx, 609)), (7, (522, 612, 550, 680))]
        text = "".join(f"{cls} {(x1+x2)/2/width:.6f} {(y1+y2)/2/height:.6f} {(x2-x1)/width:.6f} {(y2-y1)/height:.6f}\n" for cls, (x1,y1,x2,y2) in boxes)
        (labels / f"{name}.txt").write_text(text)
    classes = yaml.safe_load((ROOT / "annotation_tool/config/classes.yaml").read_text())
    names = [entry["name"] for entry in classes["classes"]]
    (output / "data.yaml").write_text(yaml.safe_dump({"path": str(output), "train": "images/train", "val": "images/train", "names": names}, sort_keys=False))
    (output / "README.txt").write_text("Synthetic geometric UI fixtures. Train and val intentionally share images: never use for evaluation.\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/demo_dataset")
    args = parser.parse_args()
    try:
        print(create_demo_dataset(args.output))
    except FileExistsError as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
