# states.py

from typing import TypedDict, List, Dict, Any, Optional


class ImageState(TypedDict, total=False):
    # ------- INPUTS / CONFIG -------

    # For single-image mode (backwards compatibility)
    image_path: str

    # ALL page images from ONE PDF manual
    image_paths: List[str]

    # Optional identifier for the manual (e.g. "BILLY_12345")
    manual_id: str

    # User request / instruction for the reasoning model
    user_query: str

    # ------- VISION LAYER (LLaVA) -------

    # Cache of base64-encoded images by path
    # { "path/to/page1.jpg": "<base64...>", ... }
    vision_raw_b64: Dict[str, str]

    # Last vision JSON object (for backward compatibility / debugging)
    vision_json: Dict[str, Any]

    # Last vision text summary for GPT (for backward compatibility)
    vision_text_for_gpt: str

    # List of per-page vision items produced by the vision agent.
    # Each item typically has:
    # {
    #   "page_index": int,
    #   "image_path": str,
    #   "vision_json": dict,
    #   "vision_text": str,
    #   "step_number": Optional[int],
    # }
    vision_items: List[Dict[str, Any]]

    # ------- REASONING LAYER (GPT) -------

    # Full answer from the reasoning model (instructions + JSON, etc.)
    gpt_answer: str

    # ------- ERROR HANDLING -------

    # If any node fails, it can set an error message here
    error: str


class IkeaState(TypedDict, total=False):


    # input
    pdf_source: Dict[str, Any]

    # preprocessing outputs
    manual_id: str                 # stable ID for DB
    title_page: Dict[str, Any]     # ref to page0 image + extracted meta (optional)
    pdf_render: Dict[str, Any]
    panel_detections: Dict[str, Any]
    step_crops: Dict[str, Any]

    # agentic fields
    current_step: int
    step_analyses: List[Dict[str, Any]]
    messages: List[Dict[str, str]]
    final_instructions: Dict[str, Any]