import base64
from io import BytesIO
from typing import Any
from PIL import Image
import json
import fitz


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
