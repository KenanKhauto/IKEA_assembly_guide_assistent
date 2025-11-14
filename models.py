import base64
import json
from io import BytesIO
from typing import TypedDict, Optional, Any

from PIL import Image
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from prompts import get_vision_system_message, VISION_USER_PROMPT, get_reasoning_system_message
from states import ImageState
from helpers import load_image_as_base64, vision_json_to_text, format_vision_list_for_gpt


# ---------- LLaVA VISION AGENT NODE ----------

def vision_llava_agent(state: ImageState) -> ImageState:
    """
    Vision agent using LLaVA (via Ollama) that:
      - reads an image
      - calls the vision model
      - returns structured JSON + a text version for the reasoning LLM
    """
    try:
        image_path = state["image_path"]

        # 1) prepare base64 image
        if "vision_raw_b64" in state and state["vision_raw_b64"]:
            image_b64 = state["vision_raw_b64"]
        else:
            image_b64 = load_image_as_base64(image_path)
            state["vision_raw_b64"] = image_b64

        image_data_url = f"data:image/jpeg;base64,{image_b64}"

        # 2) build messages for ChatOllama (LLaVA model)
        system_message = SystemMessage(content=get_vision_system_message())

        content_parts = [
            {"type": "image_url", "image_url": image_data_url},
            {"type": "text", "text": VISION_USER_PROMPT},
        ]
        human_message = HumanMessage(content=content_parts)

        # 3) call LLaVA via Ollama
        llm = ChatOllama(
            model="llava:13b", 
            temperature=0.1,
        )
        response = llm.invoke([system_message, human_message])

        raw_text = response.content

        # 4) parse JSON (robustly)
        try:
            vision_obj = json.loads(raw_text)
        except json.JSONDecodeError:
            # try to salvage common cases (e.g., model wraps JSON in markdown)
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                # remove possible "json" language tag
                cleaned = cleaned.replace("json", "", 1).strip()
            vision_obj = json.loads(cleaned)

        state["vision_json"] = vision_obj
        state["vision_text_for_gpt"] = vision_json_to_text(vision_obj)
        return state

    except Exception as e:
        state["error"] = f"Vision agent failed: {e}"
        return state


def reasoning_gpt_agent(state: ImageState) -> ImageState:
    if state.get("error"):
        return state

    # in the future, this could be a list of vision_json dicts
    # for now, if you only have one:
    vision_items = state.get("vision_items", None)
    if vision_items is None:
        # backward-compatible: single image
        single = state.get("vision_json")
        if not single:
            state["error"] = "No visual analysis available for reasoning."
            return state
        vision_items = [single]

    user_query = state.get(
        "user_query",
        "Create clear, human-readable assembly instructions based on these images.",
    )

    vision_text_block = format_vision_list_for_gpt(vision_items)

    system_msg = SystemMessage(content=get_reasoning_system_message())

    human_content = (
        "You will now receive a list of items, each representing one image from the IKEA manual.\n\n"
        "List of items (already analyzed by the vision model):\n\n"
        f"{vision_text_block}\n\n"
        "User request:\n"
        f"{user_query}"
    )
    human_msg = HumanMessage(content=human_content)

    model = ChatOllama(
        model="gpt-oss:20b",
        temperature=0.2,
    )

    response = model.invoke([system_msg, human_msg])
    state["gpt_answer"] = response.content
    return state



def main():
    # ---------- SIMPLE GRAPH WITH ONLY THE VISION AGENT ----------
    graph = StateGraph(ImageState)

    graph.add_node("vision_llava", vision_llava_agent)
    graph.add_node("reasoning_gpt", reasoning_gpt_agent)

    graph.set_entry_point("vision_llava")

    # flow: vision → reasoning → END
    graph.add_edge("vision_llava", "reasoning_gpt")
    graph.set_finish_point("reasoning_gpt")

    app = graph.compile()
    result_state = app.invoke({"image_path": "image_side2.jpg"})
    if "error" in result_state:
        print("Error:", result_state["error"])
    else:
        # print("=== Vision JSON ===")
        # print(json.dumps(result_state["vision_json"], indent=2, ensure_ascii=False))
        # print("\n=== Text for GPT ===")
        print(" ============== LLAVA Answer ==============")
        print(result_state["vision_text_for_gpt"])
        print("\n ============== GPT Reasoning Answer ==============")
        print(result_state["gpt_answer"])

if __name__ == "__main__":
    main()
