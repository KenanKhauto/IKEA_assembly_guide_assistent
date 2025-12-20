from __future__ import annotations

from typing import Any, Dict, List, Literal
from openai import OpenAI
from pathlib import Path
from states import IkeaState
from helpers import _image_to_base64


# ---------
# CONFIG
# ---------
TEXT_MODEL = "gpt-4o"
VISION_MODEL = "gpt-4o"

client = OpenAI()

# -------------------------
# LLM wrappers (replace these)
# -------------------------

def call_llm_text(messages: List[Dict[str, str]]) -> str:
    """
    messages: [{"role": "system|user|assistant", "content": "..."}]
    returns: assistant text
    """
    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content

def call_llm_vision(prompt: str, image_path: str | Path) -> str:
    """
    prompt: instruction text
    image_path: local cropped step image
    returns: assistant text (ideally JSON)
    """
    image_b64 = _image_to_base64(image_path)

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        },
                    },
                ],
            }
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content


# -------------------------
# Helpers
# -------------------------

def _get_steps(state: IkeaState) -> List[Dict[str, Any]]:
    steps = state.get("step_crops", {}).get("steps", [])
    if not isinstance(steps, list):
        raise ValueError("state['step_crops']['steps'] missing or not a list")
    return steps


# -------------------------
# Agent nodes
# -------------------------

def init_agent_state(state: IkeaState) -> Dict[str, Any]:
    _ = _get_steps(state)  # validate steps exist
    return {
        "current_step": 0,
        "step_analyses": [],
        "messages": [
            {"role": "system", "content": "You are running a 2-agent IKEA step analysis pipeline."}
        ],
    }


def instructor_agent(state: IkeaState) -> Dict[str, Any]:
    steps = _get_steps(state)
    i = int(state.get("current_step", 0))

    # If done: assemble final output
    if i >= len(steps):
        messages = list(state.get("messages", []))
        step_analyses = state.get("step_analyses", [])

        messages.append({"role": "user", "content": "All steps are analyzed. Create final ordered assembly instructions as JSON."})
        messages.append({"role": "user", "content": f"Per-step analyses:\n{step_analyses}"})

        final_text = call_llm_text(messages)
        messages.append({"role": "assistant", "content": final_text})

        return {
            "messages": messages,
            "final_instructions": {"raw": final_text},
        }

    # Otherwise: ask analyst to process current step
    step = steps[i]
    step_id = step["step_id"]

    messages = list(state.get("messages", []))
    messages.append({
        "role": "user",
        "content": (
            f"Analyze next step index={i} step_id={step_id}. "
            "Return STRICT JSON only with keys: "
            "step_id, action_summary, objects, fasteners, quantities, warnings, dependencies, confidence."
        ),
    })

    return {"messages": messages}



def step_analyst_agent(state: IkeaState) -> Dict[str, Any]:
    steps = _get_steps(state)
    i = int(state.get("current_step", 0))
    if i >= len(steps):
        return {}

    step = steps[i]
    step_id = step["step_id"]
    image_path = step["image_path"]

    prompt = (
        "You are a visual analyst for IKEA assembly instructions.\n"
        "Look at the provided step image and extract a structured description.\n"
        "Return STRICT JSON only (no markdown) with keys:\n"
        "step_id, action_summary, objects, fasteners, quantities, warnings, dependencies, confidence.\n"
        f"step_id must be '{step_id}'."
    )

    analyst_json = call_llm_vision(prompt=prompt, image_path=image_path)

    step_analyses = list(state.get("step_analyses", []))
    step_analyses.append({
        "step_id": step_id,
        "global_step_index": step.get("global_step_index"),
        "page_index": step.get("page_index"),
        "panel_index": step.get("panel_index"),
        "image_path": image_path,
        "analysis_raw": analyst_json,
    })

    messages = list(state.get("messages", []))
    messages.append({"role": "assistant", "content": f"Step analyst output for {step_id}: {analyst_json}"})

    return {
        "step_analyses": step_analyses,
        "messages": messages,
        "current_step": i + 1,
    }


# -------------------------
# Routing
# -------------------------

def route_after_crop(_: IkeaState) -> Literal["init_agent"]:
    return "init_agent"


def route_after_instructor(state: IkeaState) -> Literal["step_analyst", "__end__"]:
    if "final_instructions" in state:
        return "__end__"
    return "step_analyst"


def route_after_analyst(state: IkeaState) -> Literal["instructor", "__end__"]:
    # always return to instructor; instructor will finish when i>=len(steps)
    return "instructor"

