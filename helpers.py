import base64
from io import BytesIO
from typing import Any
from PIL import Image
import json


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
