from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
import json

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
import re


def split_rendered_steps(markdown_text: str) -> list[str]:
    # Split on "### Step X:"
    parts = re.split(r'(?=### Step \d+:)', markdown_text)
    return [p.strip() for p in parts if p.strip()]


@dataclass(frozen=True)
class LlmRenderHumanOutputNode:
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    language: str = "en"

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # your structure: final_instructions is a dict with a "raw" field
        raw_container = state.get("final_instructions") or {}
        raw = raw_container.get("raw") or raw_container  # fallback if you change later

        instructions_obj = self._coerce_to_json_obj(raw)

        llm = ChatOpenAI(model=self.model, temperature=self.temperature)

        resp = llm.invoke([
            SystemMessage(content=self._system_prompt()),
            HumanMessage(content=self._user_prompt(state, instructions_obj)),
        ])
        output_text = (resp.content or "").strip()
        return {
            "output_text": output_text,
            "output_text_list": split_rendered_steps(output_text)}

    # def _system_prompt(self) -> str:
    #     return (
    #         "You are a precise technical writer for IKEA-style assembly manuals.\n"
    #         "Write natural, helpful, human-readable instructions.\n"
    #         "Rules:\n"
    #         "- Use ONLY the provided data; do not invent parts or steps.\n"
    #         "- If something is unclear, say 'Unclear: ...' instead of guessing.\n"
    #         "- Output Markdown with sections: Parts & Fasteners, Warnings, Steps.\n"
    #     )

    def _system_prompt(self) -> str:
        return (
            "You are a precise technical writer for IKEA-style assembly instructions.\n"
            "You convert structured step JSON into human-readable, image-adjacent instructions.\n\n"
            "Hard rules:\n"
            "- Use ONLY the provided JSON content. Do not invent parts, tools, or actions.\n"
            "- Keep the same step order as in JSON.\n"
            "- Output MUST be Markdown.\n"
            "- No global 'Warnings' or global 'Parts & Fasteners' section that merges everything.\n"
            "- Warnings must appear inside the relevant step.\n"
            "- Every step must include a small 'You need' list (parts/fasteners + quantities) derived from that step.\n"
            "- If something is unclear, write: 'Unclear: ...' and move on.\n\n"
            "Style:\n"
            "- Write warm, helpful, direct instructions.\n"
            "- Add short practical micro-guidance (alignment, orientation, gentle pressure) ONLY if supported by the step data.\n"
            "- Prefer 2–5 bullet sub-steps per step. Keep it scannable.\n" \
            "- You may add brief clarification phrases (e.g., ‘make sure it sits flush’, ‘tighten evenly’) only when consistent with the given objects/fasteners and action_summary. If not supported, mark as Unclear.\n"
        )


    # def _user_prompt(self, state: Dict[str, Any], instructions_obj: Dict[str, Any]) -> str:
    #     manual_id = state.get("manual_id", "unknown_manual")
    #     return (
    #         f"Manual ID: {manual_id}\n"
    #         "Convert this JSON into a clear assembly guide.\n\n"
    #         f"{json.dumps(instructions_obj, ensure_ascii=False, indent=2)}"
    #     )

    def _user_prompt(self, state, instructions_obj) -> str:
        manual_id = state.get("manual_id", "unknown_manual")
        return (
            f"Manual ID: {manual_id}\n\n"
            "Task: Convert the JSON into step-by-step Markdown that will be shown next to each step image.\n\n"
            "Required output format:\n"
            "1) Optional short intro (1–3 lines) ONLY if helpful.\n"
            "2) Then for EACH item in assembly_instructions, output exactly:\n\n"
            "### Step {n}: {short title based on action_summary}\n"
            "**You need:**\n"
            "- part/fastener — quantity (from quantities if available; otherwise omit quantity)\n"
            "\n"
            "**What to do:**\n"
            "- 2–5 clear bullet actions, using the objects/fasteners names from JSON.\n"
            "- Mention quantities naturally (e.g., 'Use the 4 screws...') when present.\n"
            "\n"
            "**Watch out:** (ONLY if warnings exist)\n"
            "- integrate the warning(s) here.\n\n"
            "Do NOT create any global section that aggregates all parts or all warnings.\n"
            "Do NOT merge different steps together.\n"
            "Do NOT add steps that do not exist.\n\n"
            "Aim for 3–8 lines per step.\n\n"
            "Here is the JSON:\n"
            f"{json.dumps(instructions_obj, ensure_ascii=False, indent=2)}"
        )


    def _coerce_to_json_obj(self, raw: Any) -> Dict[str, Any]:
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            s = raw.strip()
            if s.startswith("```"):
                s = self._strip_code_fence(s)
            return json.loads(s)
        raise TypeError(f"Expected dict or JSON string, got {type(raw).__name__}")

    def _strip_code_fence(self, s: str) -> str:
        lines = s.splitlines()
        # remove first fence and last fence if present
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
