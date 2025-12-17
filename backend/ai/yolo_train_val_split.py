from pathlib import Path
import random
import shutil

# ---- CONFIG ----
SOURCE_DATASET = Path(r"C:\Users\Kenan\Desktop\IKEA_Project\IKEA_assembly_guide_assistent\dataset")
OUT_DATASET    = Path(r"C:\Users\Kenan\Desktop\IKEA_Project\IKEA_assembly_guide_assistent\dataset_yolo")

TRAIN_RATIO = 0.8
SEED = 42

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def extract_hash6(filename_stem: str) -> str:
    """
    From: Chair__hemnes__a3f92c__page_012
    Return: a3f92c
    """
    parts = filename_stem.split("__")
    if len(parts) < 4:
        raise ValueError(f"Filename doesn't match expected pattern: {filename_stem}")
    return parts[2]


def main():
    images_dir = SOURCE_DATASET / "images"
    labels_dir = SOURCE_DATASET / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Missing: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Missing: {labels_dir}")

    # Create output dirs
    (OUT_DATASET / "images" / "train").mkdir(parents=True, exist_ok=True)
    (OUT_DATASET / "images" / "val").mkdir(parents=True, exist_ok=True)
    (OUT_DATASET / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (OUT_DATASET / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # Collect images and group by pdf hash
    image_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    if not image_paths:
        raise RuntimeError(f"No images found in: {images_dir}")

    groups = {}  # hash6 -> list[Path]
    for img_path in image_paths:
        h = extract_hash6(img_path.stem)
        groups.setdefault(h, []).append(img_path)

    hashes = sorted(groups.keys())
    rng = random.Random(SEED)
    rng.shuffle(hashes)

    n_train = int(len(hashes) * TRAIN_RATIO)
    train_hashes = set(hashes[:n_train])
    val_hashes   = set(hashes[n_train:])

    print(f"Total manuals (hash groups): {len(hashes)}")
    print(f"Train manuals: {len(train_hashes)} | Val manuals: {len(val_hashes)}")
    print(f"Total images: {len(image_paths)}")

    copied_images = {"train": 0, "val": 0}
    copied_labels = {"train": 0, "val": 0}
    missing_labels = 0

    # Copy files
    for h, imgs in groups.items():
        split = "train" if h in train_hashes else "val"

        for img_path in imgs:
            # copy image
            dst_img = OUT_DATASET / "images" / split / img_path.name
            shutil.copy2(img_path, dst_img)
            copied_images[split] += 1

            # copy label if exists (negatives might have none)
            lab_path = labels_dir / (img_path.stem + ".txt")
            if lab_path.exists():
                dst_lab = OUT_DATASET / "labels" / split / lab_path.name
                shutil.copy2(lab_path, dst_lab)
                copied_labels[split] += 1
            else:
                missing_labels += 1

    print("\nDone.")
    print(f"Images copied  - train: {copied_images['train']} | val: {copied_images['val']}")
    print(f"Labels copied  - train: {copied_labels['train']} | val: {copied_labels['val']}")
    print(f"Images w/ no label file (negatives): {missing_labels}")
    print(f"Output folder: {OUT_DATASET}")


if __name__ == "__main__":
    main()
