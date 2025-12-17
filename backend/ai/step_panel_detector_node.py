from __future__ import annotations
from typing import Dict, Any, List

class DetectStepPanelsNode:
    def __init__(self, detector):
        self.detector = detector

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pdf_render = state.get("pdf_render")
        if not pdf_render or "pages" not in pdf_render:
            raise ValueError("Missing state['pdf_render']['pages']")

        img_paths: List[str] = []
        page_indices: List[int] = []

        for p in pdf_render["pages"]:
            if "image_path" not in p:
                continue  # skip failed renders
            img_paths.append(p["image_path"])
            page_indices.append(int(p["page_index"]))

        detections_per_page = self.detector.predict_many(img_paths)

        per_page = {}  # page_index -> list of det dicts
        for page_index, dets in zip(page_indices, detections_per_page):
            per_page[page_index] = []
            for d in dets:
                per_page[page_index].append({
                    "bbox_xyxy": d.bbox_xyxy,
                    "confidence": d.conf,
                    "class_id": d.cls,
                })

        return {
            "panel_detections": {
                "by_page": per_page,
                "conf": self.detector.conf,
                "iou": self.detector.iou,
            }
        }
