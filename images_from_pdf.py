from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF
import cv2
import numpy as np


# ---------- STEP-NUMBER DETECTION ----------

def detect_step_number_boxes(
    img: np.ndarray,
    min_area_ratio: float = 0.002,       # minimal digit area vs page area
    min_height_ratio: float = 0.035,     # minimal digit height vs page height
    max_width_height_ratio: float = 1.2, # digits should not be very wide
    max_x_ratio: float = 0.25,           # must be in left quarter of the page
) -> List[Tuple[int, int, int, int]]:
    """
    Detect bounding boxes (x, y, w, h) for large step numbers on the page.
    """
    h, w = img.shape[:2]
    img_area = w * h

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # IKEA pages: white background, black drawings
    _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[Tuple[int, int, int, int]] = []

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh

        # filter tiny stuff
        if area < img_area * min_area_ratio:
            continue
        if bh < h * min_height_ratio:
            continue
        # must be near left side
        if x > w * max_x_ratio:
            continue
        # digit-ish: not super wide
        if bw / bh > max_width_height_ratio:
            continue

        boxes.append((x, y, bw, bh))

    # top-to-bottom
    boxes.sort(key=lambda b: b[1])
    return boxes


def crop_steps_by_numbers(
    page_img: np.ndarray,
    page_name_stem: str,
    output_dir: Path,
    top_margin_ratio: float = 0.01,
    bottom_margin_ratio: float = 0.01,
) -> List[Path]:
    """
    Take a page image and crop it into one image per IKEA step,
    using step numbers as vertical anchors.
    """
    H, W = page_img.shape[:2]

    boxes = detect_step_number_boxes(page_img)

    output_dir.mkdir(parents=True, exist_ok=True)

    # if no numbers -> treat whole page as one "step"
    if not boxes:
        out_path = output_dir / f"{page_name_stem}_step_01.png"
        cv2.imwrite(str(out_path), page_img)
        return [out_path]

    step_paths: List[Path] = []

    top_margin = int(H * top_margin_ratio)
    bottom_margin = int(H * bottom_margin_ratio)

    for i, (_, y, _, bh) in enumerate(boxes):
        top = max(0, y - top_margin)
        if i + 1 < len(boxes):
            next_y = boxes[i + 1][1]
            bottom = max(top + 10, next_y - bottom_margin)
        else:
            bottom = H

        crop = page_img[top:bottom, :]
        out_path = output_dir / f"{page_name_stem}_step_{i + 1:02d}.png"
        cv2.imwrite(str(out_path), crop)
        step_paths.append(out_path)

    return step_paths


# ---------- PDF → PAGE IMAGES → STEP IMAGES ----------

def pdf_to_step_images(
    pdf_path: str | Path,
    steps_root_dir: str | Path,
    dpi: int = 300,
) -> List[Path]:
    """
    High-level helper:
      1. Render each page of the PDF to an image.
      2. On each page, detect step numbers.
      3. Crop vertically between numbers to get one image per step.

    Returns a list of all step image paths.
    """
    pdf_path = Path(pdf_path)
    steps_root_dir = Path(steps_root_dir)
    steps_root_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    print(f"[PDF] Opened {pdf_path}, pages: {len(doc)}")

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    all_step_paths: List[Path] = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        pix = page.get_pixmap(matrix=mat)

        # render page into a NumPy image (BGR for OpenCV)
        img = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img.reshape(pix.height, pix.width, pix.n)  # n = number of channels
        if pix.n == 4:
            # convert RGBA to BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        page_name_stem = f"page_{page_index + 1:03d}"
        page_steps_dir = steps_root_dir / page_name_stem

        step_paths = crop_steps_by_numbers(
            page_img=img,
            page_name_stem=page_name_stem,
            output_dir=page_steps_dir,
        )

        print(f"[PAGE {page_index + 1}] extracted {len(step_paths)} step image(s).")
        all_step_paths.extend(step_paths)

    print(f"[ALL] Total step images: {len(all_step_paths)}")
    return all_step_paths


if __name__ == "__main__":
    # Example usage
    pdf_file = ".\pdfs\ikea2.pdf"   
    steps_dir = "out_steps"           # will contain page_xxx/page_xxx_step_yy.png

    pdf_to_step_images(pdf_file, steps_dir, dpi=300)
