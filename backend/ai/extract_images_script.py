from pathlib import Path
import random
from pdf_to_imgs_node import PdfToImagesNode
import shutil

# your node
# from your_module.pdf_to_images_node import PdfToImagesNode

PDF_ROOT = Path(r"C:\Users\Kenan\Desktop\harvord_ikea\pdfs")
ARTIFACTS_ROOT = Path("artifacts")
DATASET_IMAGES = Path(r"dataset\images")

PDFS_PER_CLASS = 3
SEED = 42
DPI_DEFAULT = 200



def collect_pdfs_per_class(pdf_root: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for class_dir in pdf_root.iterdir():
        if not class_dir.is_dir():
            continue
        pdfs = list(class_dir.rglob("*.pdf"))
        if pdfs:
            out[class_dir.name] = pdfs
    return out


def sample_per_class(pdfs_per_class: dict[str, list[Path]], n: int, seed: int) -> dict[str, list[Path]]:
    rng = random.Random(seed)
    sampled: dict[str, list[Path]] = {}
    for cls, pdfs in pdfs_per_class.items():
        k = min(n, len(pdfs))
        sampled[cls] = rng.sample(pdfs, k)
    return sampled


def safe_name(s: str) -> str:
    """Make folder names safe for filenames."""
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)


def copy_pages_to_dataset(class_name: str, product_name: str, cache_key: str) -> int:
    """
    Copies all page_*.png from artifacts/pdf_pages/<cache_key> to dataset/images
    with unique names: <class>__<product>__<pdfhash6>__page_XXX.png
    Returns count copied.
    """
    src_dir = ARTIFACTS_ROOT / "pdf_pages" / cache_key
    if not src_dir.exists():
        raise FileNotFoundError(f"Artifact dir not found: {src_dir}")

    DATASET_IMAGES.mkdir(parents=True, exist_ok=True)

    pdf_hash_short = cache_key.split(":")[0][:6]
    cls = safe_name(class_name)
    prod = safe_name(product_name)

    copied = 0
    for img_path in sorted(src_dir.glob("page_*.png")):
        new_name = f"{cls}__{prod}__{pdf_hash_short}__{img_path.name}"
        dst_path = DATASET_IMAGES / new_name

        # Avoid overwriting anything (if you re-run, you can choose to skip instead)
        if dst_path.exists():
            # If you prefer skipping instead of erroring, replace this with `continue`
            raise RuntimeError(f"Collision detected: {dst_path}")

        shutil.copy2(img_path, dst_path)
        copied += 1

    return copied


def main():
    # 1) collect + sample PDFs
    pdfs_per_class = collect_pdfs_per_class(PDF_ROOT)
    sampled = sample_per_class(pdfs_per_class, PDFS_PER_CLASS, SEED)

    # 2) init node once
    node = PdfToImagesNode(artifact_root=ARTIFACTS_ROOT, dpi=DPI_DEFAULT)

    total_pdfs = sum(len(v) for v in sampled.values())
    done = 0

    for cls, pdf_list in sampled.items():
        print(f"\n=== Class: {cls} (sampling {len(pdf_list)}) ===")
        for pdf_path in pdf_list:
            done += 1
            product_name = pdf_path.parent.name  # product folder

            # 3) run PDF -> images
            state = {
                "pdf_source": {
                    "kind": "path",
                    "path": str(pdf_path),
                    "filename": pdf_path.name,
                },
                # optional per-run overrides:
                # "pdf_to_images_config": {"dpi": 200, "cache_policy": "use_cache"}
            }

            out = node(state)
            render = out["pdf_render"]
            cache_key = render["cache_key"]

            # 4) copy + rename into dataset/images
            copied = copy_pages_to_dataset(cls, product_name, cache_key)

            print(f"[{done}/{total_pdfs}] {pdf_path}")
            print(f"  pages rendered: {render['page_count']}")
            print(f"  copied to dataset/images: {copied}")
            print(f"  cache_key: {cache_key}")


if __name__ == "__main__":
    main()