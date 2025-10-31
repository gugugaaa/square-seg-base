import argparse
import json
from pathlib import Path
from typing import Iterable, List

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate empty COCO polygon annotations from images.")
    parser.add_argument("--images", required=True, type=Path, help="Path to the directory containing images.")
    parser.add_argument(
        "--labels",
        required=True,
        type=Path,
        help="Path to the output COCO JSON (e.g., instances_val.json).",
    )
    return parser.parse_args()


def find_image_files(root: Path) -> List[Path]:
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted(
        [
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in image_exts
        ]
    )


def build_image_entry(idx: int, image_path: Path) -> dict:
    with Image.open(image_path) as img:
        width, height = img.size
    return {
        "id": idx,
        "file_name": image_path.name,
        "width": width,
        "height": height,
    }


def build_dataset(images: Iterable[Path]) -> dict:
    coco_images = [build_image_entry(i + 1, path) for i, path in enumerate(images)]
    return {
        "info": {"description": "Auto-generated COCO dataset", "version": "1.0", "year": 2024},
        "licenses": [],
        "images": coco_images,
        "annotations": [],
        "categories": [
            {"id": 0, "name": "square", "supercategory": "shape"},
        ],
    }


def main() -> None:
    args = parse_args()
    image_dir = args.images
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    images = find_image_files(image_dir)
    dataset = build_dataset(images)
    output_path = args.labels
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(dataset, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
