"""
organize_dataset.py
--------------------
Turns a raw downloaded dataset (e.g. AI-ArtBench, which ships as many
style-specific folders like "AI_LD_baroque", "AI_SD_realism", "real_ukiyo_e")
into the clean binary structure our training script expects:

    dataset/
        train/
            ai_generated/
            real_painting/
        validation/
            ai_generated/
            real_painting/
        test/
            ai_generated/
            real_painting/

Use --max-per-class to cap how many images get used per class, so training
finishes in a reasonable time on a laptop CPU.

Usage:
    python utils/organize_dataset.py --source raw_dataset --max-per-class 6000 --symlink
"""

import argparse
import random
import shutil
from pathlib import Path

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def is_ai_folder(folder_name: str) -> bool:
    name = folder_name.lower()
    return "ai" in name or "_ld_" in name or "_sd_" in name


def collect_images(source_dir: Path):
    ai_images, real_images = [], []
    for path in source_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            parent_name = path.parent.name
            if is_ai_folder(parent_name):
                ai_images.append(path)
            else:
                real_images.append(path)
    return ai_images, real_images


def split_and_copy(images, class_name: str, dest_root: Path, link_instead_of_copy: bool, max_per_class):
    random.seed(SEED)
    random.shuffle(images)

    if max_per_class is not None:
        images = images[:max_per_class]

    n = len(images)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    splits = {
        "train": images[:n_train],
        "validation": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    for split_name, split_images in splits.items():
        out_dir = dest_root / split_name / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for img_path in split_images:
            dest_path = out_dir / f"{img_path.parent.name}_{img_path.name}"
            if dest_path.exists():
                continue
            if link_instead_of_copy:
                dest_path.symlink_to(img_path.resolve())
            else:
                shutil.copy2(img_path, dest_path)

    print(f"  {class_name}: {n_train} train / {len(splits['validation'])} val "
          f"/ {len(splits['test'])} test (total {n})")


def main():
    parser = argparse.ArgumentParser(description="Organize a raw art dataset into binary train/val/test folders.")
    parser.add_argument("--source", required=True, help="Path to the raw, already-downloaded dataset folder")
    parser.add_argument("--dest", default="dataset", help="Destination root (default: dataset/)")
    parser.add_argument("--symlink", action="store_true", help="Symlink instead of copy, to save disk space")
    parser.add_argument("--max-per-class", type=int, default=None,
                         help="Cap the number of images used per class (e.g. 6000).")
    args = parser.parse_args()

    source_dir = Path(args.source)
    dest_root = Path(args.dest)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source folder not found: {source_dir}")

    print(f"Scanning {source_dir} for images...")
    ai_images, real_images = collect_images(source_dir)
    print(f"Found {len(ai_images)} AI-generated images and {len(real_images)} real images.\n")

    if not ai_images or not real_images:
        raise RuntimeError(
            "Couldn't confidently classify folders into AI vs real. "
            "Open utils/organize_dataset.py and adjust is_ai_folder() to match your folder names."
        )

    if args.max_per_class:
        print(f"Capping at {args.max_per_class} images per class before splitting.\n")

    print("Splitting into train/validation/test (70/15/15)...")
    split_and_copy(ai_images, "ai_generated", dest_root, args.symlink, args.max_per_class)
    split_and_copy(real_images, "real_painting", dest_root, args.symlink, args.max_per_class)

    print(f"\nDone. Dataset organized under: {dest_root.resolve()}")


if __name__ == "__main__":
    main()