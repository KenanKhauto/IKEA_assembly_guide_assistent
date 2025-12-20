from langgraph.graph import StateGraph, END

from states import IkeaState

# NEW agent nodes + routing from the updated pipeline
from agent_nodes import (
    init_agent_state,
    instructor_agent,
    step_analyst_agent,
    route_after_crop,
    route_after_instructor,
    route_after_analyst,
)


def build_ikea_full_graph(
    pdf_to_images_node,          # callable(state) -> dict
    detect_step_panels_node,     # callable(state) -> dict
    crop_step_panels_node,       # callable(state) -> dict
):
    g = StateGraph(IkeaState)

    # preprocessing nodes
    g.add_node("pdf_to_images", pdf_to_images_node)
    g.add_node("detect_step_panels", detect_step_panels_node)
    g.add_node("crop_step_panels", crop_step_panels_node)

    # agent nodes
    g.add_node("init_agent", init_agent_state)
    g.add_node("instructor", instructor_agent)
    g.add_node("step_analyst", step_analyst_agent)

    # linear preprocess
    g.set_entry_point("pdf_to_images")
    g.add_edge("pdf_to_images", "detect_step_panels")
    g.add_edge("detect_step_panels", "crop_step_panels")
    g.add_conditional_edges("crop_step_panels", route_after_crop, {"init_agent": "init_agent"})

    # agent loop
    g.add_edge("init_agent", "instructor")
    g.add_conditional_edges("instructor", route_after_instructor, {
        "step_analyst": "step_analyst",
        "__end__": END,
    })
    g.add_conditional_edges("step_analyst", route_after_analyst, {
        "instructor": "instructor",
        "__end__": END,
    })

    return g.compile()