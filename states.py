

from typing import TypedDict, Optional, Any


# ---------- STATE ----------

class ImageState(TypedDict, total=False):
    image_path: str                         # input
    vision_raw_b64: str                     # optional cached base64
    vision_json: dict[str, Any]             # structured vision output
    vision_text_for_gpt: str                # nicely formatted text for the reasoning LLM
    user_query: str                         # input question from user  
    gpt_answer: str                         # final answer from GPT reasoning model
    error: Optional[str]                    # error message, if any

