from pathlib import Path
import shutil

ARTIFACTS_ROOT = Path(r"\artifacts\pdf_pages")
DATASET_IMAGES = Path(r"dataset\images")

DATASET_IMAGES.mkdir(parents=True, exist_ok=True)


def copy_pages_to_dataset(
    class_name: str,
    product_name: str,
    cache_key: str,
):
    src_dir = ARTIFACTS_ROOT / cache_key
    if not src_dir.exists():
        raise FileNotFoundError(f"Artifact dir not found: {src_dir}")

    pdf_hash_short = cache_key.split(":")[0][:6]

    for img_path in sorted(src_dir.glob("page_*.png")):
        new_name = f"{class_name}__{product_name}__{pdf_hash_short}__{img_path.name}"
        dst_path = DATASET_IMAGES / new_name

        if dst_path.exists():
            raise RuntimeError(f"Collision detected: {dst_path}")

        shutil.copy2(img_path, dst_path)


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Example values — plug these in from your sampler output
    copy_pages_to_dataset(
        class_name="Chair",
        product_name="hemnes",
        cache_key="a3f92c8e4d7b...:dpi200:png:pymupdf_v1"
    )
