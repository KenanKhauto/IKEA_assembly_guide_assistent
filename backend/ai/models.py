import json

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from prompts import get_vision_system_message, VISION_USER_PROMPT, get_reasoning_system_message
from states import ImageState
from helpers import load_image_as_base64, vision_json_to_text, format_vision_list_for_gpt, build_previous_pages_context, _parse_vision_json


def vision_llava_agent(state: ImageState) -> ImageState:
    """
    Vision agent using LLaVA (via Ollama) that:

      - reads ALL page images from ONE IKEA PDF manual
      - calls the vision model once per page
      - produces a list of `vision_items` for downstream reasoning

    Each entry in `vision_items` is a dict like:
      {
        "page_index": int,
        "image_path": str,
        "vision_json": dict,    # structured output from LLaVA
        "vision_text": str,     # human-readable text for GPT
        # optional:
        # "step_number": int
      }
    """
    try:
        image_paths = state.get("image_paths")
        if not image_paths:
            # backward-compatible fallback: single image_path
            single_path = state.get("image_path")
            if not single_path:
                state["error"] = "No image_paths or image_path provided to vision_llava_agent."
                return state
            image_paths = [single_path]

        # prepare model and system message once
        llm = ChatOllama(
            model="llava:13b",
            temperature=0.1,
        )
        system_message = SystemMessage(content=get_vision_system_message())

        vision_items = []
        raw_b64_cache = state.get("vision_raw_b64") or {}

        for page_index, image_path in enumerate(image_paths):
            # 1) base64 for this image (cached if available)
            if image_path in raw_b64_cache:
                image_b64 = raw_b64_cache[image_path]
            else:
                image_b64 = load_image_as_base64(image_path)
                raw_b64_cache[image_path] = image_b64

            image_data_url = f"data:image/jpeg;base64,{image_b64}"

            # 2) message with this page only
            content_parts = [
                {"type": "image_url", "image_url": image_data_url},
                {"type": "text", "text": VISION_USER_PROMPT},
            ]
            human_message = HumanMessage(content=content_parts)

            # 3) call LLaVA
            response = llm.invoke([system_message, human_message])
            raw_text = response.content

            # 4) parse JSON
            vision_json = _parse_vision_json(raw_text)

            # 5) convert to human-readable text for reasoning model
            vision_text = vision_json_to_text(vision_json)

            vision_item = {
                "page_index": page_index,
                "image_path": image_path,
                "vision_json": vision_json,
                "vision_text": vision_text,
            }

            # if the JSON already contains a step number, keep it
            step_number = vision_json.get("step_number") or vision_json.get("step")
            if step_number is not None:
                vision_item["step_number"] = step_number

            vision_items.append(vision_item)

        # store results in state
        state["vision_items"] = vision_items
        state["vision_raw_b64"] = raw_b64_cache

        # optional: keep last one for backward compatibility
        if vision_items:
            state["vision_json"] = vision_items[-1]["vision_json"]
            state["vision_text_for_gpt"] = vision_items[-1]["vision_text"]

        return state

    except Exception as e:
        state["error"] = f"Vision agent failed: {e}"
        return state


def reasoning_gpt_agent(state: ImageState) -> ImageState:
    """
    Reasoning agent using a GPT-style model (gpt-oss:20b via Ollama) that:

      - receives ALL `vision_items` for one IKEA manual
      - connects the dots across pages/steps
      - produces global, ordered assembly instructions

    It does *no* vision – it only works with the structured summaries from LLaVA.
    """
    if state.get("error"):
        return state

    vision_items = state.get("vision_items")
    if not vision_items:
        state["error"] = "No visual analysis available for reasoning (vision_items is empty)."
        return state

    user_query = state.get(
        "user_query",
        "Create clear, human-readable assembly instructions for this entire IKEA manual.",
    )

    # convert the list of per-page items to a text block for the LLM
    # you can decide what format this uses in `format_vision_list_for_gpt`
    vision_text_block = format_vision_list_for_gpt(vision_items)

    system_msg = SystemMessage(content=get_reasoning_system_message())

    # prompt explicitly describing the task
    human_content = (
        "You are given a complete IKEA furniture manual, already analyzed page-by-page by a vision model.\n"
        "Each item below represents ONE page (or step) with structured information about parts, actions, and notes.\n\n"
        "Your tasks:\n"
        "1. Infer the correct global order of assembly steps.\n"
        "2. Merge pages that belong to the same logical step if necessary.\n"
        "3. Produce:\n"
        "   a) Clear, human-readable step-by-step assembly instructions in good English.\n"
        "   b) A machine-readable JSON structure with the following shape:\n"
        "      {\n"
        "        \"steps\": [\n"
        "          {\n"
        "            \"step_number\": int,\n"
        "            \"title\": str,\n"
        "            \"description\": str,\n"
        "            \"required_parts\": [ {\"id\": str, \"count\": int, \"type\": str } ],\n"
        "            \"tools\": [str],\n"
        "            \"notes\": [str]\n"
        "          },\n"
        "          ...\n"
        "        ]\n"
        "      }\n\n"
        "Important:\n"
        "- Respect the part identifiers and quantities from the input as much as possible.\n"
        "- If information is ambiguous or missing, make a reasonable assumption and clearly mark it as such.\n"
        "- The natural language instructions should be easy to follow for a non-technical user.\n\n"
        "----- BEGIN VISION ITEMS -----\n"
        f"{vision_text_block}\n"
        "----- END VISION ITEMS -----\n\n"
        f"User request: {user_query}\n"
        "Return BOTH the human-readable instructions and the JSON in your answer."
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
    graph = StateGraph(ImageState)

    graph.add_node("vision_llava", vision_llava_agent)
    graph.add_node("reasoning_gpt", reasoning_gpt_agent)

    graph.set_entry_point("vision_llava")
    graph.add_edge("vision_llava", "reasoning_gpt")
    graph.set_finish_point("reasoning_gpt")

    app = graph.compile()

    result_state = app.invoke({
        "image_paths": [
           
            r".\images\2.jpg",
            r".\images\3.jpg",
            r".\images\4.jpg",
            # ...
        ],
    })

    if "error" in result_state:
        print("Error:", result_state["error"])
    else:
        print(" ============== GPT Reasoning Answer ==============")
        print(result_state["gpt_answer"])


if __name__ == "__main__":
    main()
