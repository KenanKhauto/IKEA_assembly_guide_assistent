from typing import Dict, Any
from yolo_step_panel_detector import StepPanelDetector

class DetectStepPanelsNode:
    def __init__(self, detector: StepPanelDetector):
        self.detector = detector

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pdf_render = state.get("pdf_render")
        if not pdf_render or "pages" not in pdf_render:
            raise ValueError("Missing state['pdf_render']['pages']")

        pages = pdf_render["pages"]

        # Collect image paths in the same order
        img_paths = []
        page_indices = []
        for p in pages:
            # skip failed renders
            if "image_path" not in p:
                continue
            img_paths.append(p["image_path"])
            page_indices.append(p["page_index"])

        detections_per_page = self.detector.predict_many(img_paths)

        step_candidates = []
        for page_index, dets in zip(page_indices, detections_per_page):
            for d in dets:
                step_candidates.append({
                    "page_index": page_index,              # 0-based
                    "bbox_xyxy": d.bbox_xyxy,              # pixel coords
                    "confidence": d.conf,
                    "class_id": d.cls,
                })

        return {"step_candidates": step_candidates}
