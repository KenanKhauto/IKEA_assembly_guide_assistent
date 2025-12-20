from __future__ import annotations
import base64
from io import BytesIO
from typing import Any
from PIL import Image
import json
import fitz
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pathlib import Path
import re
from typing import List, Tuple

# ----------------------------------
# Project root
# ----------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts_test"
STEP_CROPS_DIR = ARTIFACTS_DIR / "step_crops"

# ----------------------------------
# Step image parsing
# ----------------------------------

STEP_RE = re.compile(
    r"step_(?P<pdfid>[a-f0-9]+)_p(?P<page>\d+)_s(?P<step>\d+)\.png$",
    re.IGNORECASE
)

def collect_sorted_step_images(pdf_id: str) -> List[Path]:
    """
    Collects and sorts step images for one PDF by (page, step).

    Returns absolute Paths.
    """
    pdf_dir = STEP_CROPS_DIR / pdf_id

    if not pdf_dir.exists():
        raise FileNotFoundError(f"Step crop directory not found: {pdf_dir}")

    items: List[Tuple[int, int, Path]] = []

    for img in pdf_dir.iterdir():
        m = STEP_RE.match(img.name)
        if not m:
            continue

        page = int(m.group("page"))
        step = int(m.group("step"))
        items.append((page, step, img))

    if not items:
        raise RuntimeError(f"No step images found in {pdf_dir}")

    items.sort(key=lambda x: (x[0], x[1]))
    return [p for _, __, p in items]

def _image_to_base64(image_path: str | Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def pdf_to_b64_images(pdf_path, dpi=220):
    imgs = []
    doc = fitz.open(pdf_path)
    try:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            png_bytes = pix.tobytes("png")
            imgs.append(base64.b64encode(png_bytes).decode("utf-8"))
    finally:
        doc.close()
    return imgs


def load_image_as_base64(path: str) -> str:
    """Load image from disk and return JPEG base64 string."""
    with Image.open(path) as img:
        # Convert everything to JPEG for simplicity
        buf = BytesIO()
        img.convert("RGB").save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def vision_json_to_text(vision: dict[str, Any]) -> str:
    """
    Turn the structured JSON into a readable text block
    that can be directly fed into the reasoning LLM.
    """
    parts: list[str] = []

    parts.append(f"Scene summary: {vision.get('scene_summary', '')}\n")

    entities = vision.get("entities", []) or []
    if entities:
        parts.append("Entities:")
        for e in entities:
            name = e.get("name", "")
            cat = e.get("category", "")
            attrs = ", ".join(e.get("attributes", []) or [])
            pos = e.get("approx_position", "")
            parts.append(f"  - {name} [{cat}], attributes: {attrs}, position: {pos}")
        parts.append("")

    relationships = vision.get("relationships", []) or []
    if relationships:
        parts.append("Relationships:")
        for r in relationships:
            parts.append(f"  - {r}")
        parts.append("")

    actions = vision.get("actions", []) or []
    if actions:
        parts.append("Actions:")
        for a in actions:
            parts.append(f"  - {a}")
        parts.append("")

    text_items = vision.get("text_in_image", []) or []
    if text_items:
        parts.append("Text in image:")
        for t in text_items:
            txt = t.get("text", "")
            loc = t.get("location", "")
            style = t.get("style", "")
            parts.append(f"  - \"{txt}\" at {loc} ({style})")
        parts.append("")

    style = vision.get("style", {}) or {}
    if style:
        parts.append(
            "Visual style: "
            f"medium={style.get('medium','')}, "
            f"lighting={style.get('lighting','')}, "
            f"camera={style.get('camera','')}"
        )
        parts.append("")

    salient = vision.get("salient_details_for_reasoning", []) or []
    if salient:
        parts.append("Salient details for reasoning:")
        for s in salient:
            parts.append(f"  - {s}")
        parts.append("")

    return "\n".join(parts).strip()


def format_vision_list_for_gpt(vision_items: list[dict]) -> str:
    lines = []
    for idx, item in enumerate(vision_items, start=1):
        role = item.get("image_role", "unknown")
        step = item.get("step_number", "")
        summary = item.get("scene_summary", "")

        lines.append(f"=== ITEM {idx} ===")
        lines.append(f"image_role: {role}")
        if step:
            lines.append(f"step_number: {step}")
        lines.append("full_json:")
        lines.append(json.dumps(item, ensure_ascii=False, indent=2))
        lines.append("")  # blank line

    return "\n".join(lines)


def build_previous_pages_context(vision_items: list[dict], max_pages: int = 3) -> str:
    """
    Build a short context string summarizing previous pages
    to give LLaVA some memory without blowing up the context length.
    """
    # only take the last `max_pages` pages to keep it short
    subset = vision_items[-max_pages:]
    parts = []
    for item in subset:
        page = item.get("page_index")
        text = item.get("raw_text")
        parts.append(f"- Page {page}: {text}")
    return "\n".join(parts)


def _parse_vision_json(raw_text: str) -> dict:
    """
    Robust JSON parsing: handles plain JSON and markdown ```json``` blocks.
    """
    raw_text = raw_text.strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        # try to clean markdown fencing
        cleaned = raw_text
        if cleaned.startswith("```"):
            # strip leading/trailing backticks
            cleaned = cleaned.strip("`")
            # remove a possible 'json' language tag at the top
            if cleaned.lstrip().lower().startswith("json"):
                cleaned = cleaned.lstrip()[4:].strip()
        return json.loads(cleaned)
    

def format_vision_list_for_gpt(vision_items: list[dict]) -> str:
    """
    Turn vision_items into a readable list for the reasoning LLM.
    Assumes each item has 'page_index' and 'vision_text'.
    """
    lines = []
    # sort by page_index just to be safe
    for item in sorted(vision_items, key=lambda x: x.get("page_index", 0)):
        page = item.get("page_index")
        step = item.get("step_number", "unknown")
        text = item.get("raw_text")
        lines.append(f"[PAGE {page}, STEP {step}]\n{text}\n")
    return "\n".join(lines)




def visualize_detections(
    image_path: Union[str, Path],
    detections: Sequence[Any],
    title: Optional[str] = None,
    show_conf: bool = True,
    linewidth: int = 2,
) -> None:
    """
    Draw YOLO detections on an image and display it.

    detections can be:
      - List[Detection] where each has .bbox_xyxy and .conf
      - List[dict] where each has "bbox_xyxy" (or "bbox") and optionally "confidence"
      - Any object with bbox in xyxy form
    """
    img = Image.open(image_path).convert("RGB")

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title or Path(image_path).name)

    def _get_bbox_xyxy(d: Any) -> List[float]:
        # dataclass Detection
        if hasattr(d, "bbox_xyxy"):
            return list(getattr(d, "bbox_xyxy"))
        # dict styles
        if isinstance(d, dict):
            if "bbox_xyxy" in d:
                return list(d["bbox_xyxy"])
            if "bbox" in d:
                return list(d["bbox"])
        raise ValueError("Detection format not recognized. Expected bbox_xyxy.")

    def _get_conf(d: Any) -> Optional[float]:
        if hasattr(d, "conf"):
            return float(getattr(d, "conf"))
        if isinstance(d, dict) and "confidence" in d:
            return float(d["confidence"])
        return None

    for det in detections:
        x1, y1, x2, y2 = _get_bbox_xyxy(det)
        w = x2 - x1
        h = y2 - y1

        rect = patches.Rectangle((x1, y1), w, h, fill=False, linewidth=linewidth)
        ax.add_patch(rect)

        if show_conf:
            conf = _get_conf(det)
            if conf is not None:
                ax.text(
                    x1,
                    max(0, y1 - 5),
                    f"{conf:.2f}",
                    fontsize=10,
                    va="bottom",
                )

    plt.show()