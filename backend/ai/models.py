from dotenv import load_dotenv
from pathlib import Path
from pdf_to_imgs_node import PdfToImagesNode
from step_panel_detector_node import DetectStepPanelsNode
from yolo_step_panel_detector import StepPanelDetector
from crop_step_panel_node import CropStepPanelsNode
from final_graph import build_ikea_full_graph



def use_agents(path_pdfnode, path_detectornode, path_cropnode, path_to_pdf):
    pdf_node = PdfToImagesNode(
    artifact_root=Path(path_pdfnode),
    dpi=200,
    )

    load_dotenv()
    # --- YOLO detector
    detector = StepPanelDetector(
        weights_path=path_detectornode,
        conf=0.20,
        iou=0.7,
        device="cuda:0",
    )

    detect_node = DetectStepPanelsNode(detector)

    # --- crop node
    crop_node = CropStepPanelsNode(
        artifacts_root=Path(path_cropnode),
        padding_px=16,
        cache_policy="use_cache",
    )

    graph = build_ikea_full_graph(pdf_node, detect_node, crop_node)
    # r"C:\Users\Kenan\Desktop\harvord_ikea\pdfs\Shelf\pinnig\0.pdf"
    state0 = {
        "pdf_source": {"kind": "path", "path": path_to_pdf, "filename": "ikea4.pdf"}
    }

    final_state = graph.invoke(state0, config={"recursion_limit": 200})

    return graph, final_state