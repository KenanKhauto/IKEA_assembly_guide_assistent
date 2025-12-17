from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import numpy as np
from ultralytics import YOLO
import os


@dataclass
class Detection:
    bbox_xyxy: List[float]   # [x1, y1, x2, y2] in pixels
    conf: float
    cls: int


class StepPanelDetector:
    """
    YOLOv8 inference wrapper for step_panel detection.
    Loads the model once and can run on image paths.
    """

    def __init__(
        self,
        weights_path: Union[str, Path],
        conf: float = 0.25,
        iou: float = 0.7,
        device: Optional[Union[str, int]] = None,  # "cuda:0", 0, "cpu"
    ):
        self.weights_path = str(weights_path)
        self.conf = conf
        self.iou = iou
        self.device = device
        self.model = YOLO(self.weights_path)

    def predict_image(self, image_path: Union[str, Path]) -> List[Detection]:
        """
        Returns a list of detections for a single image.
        """
        results = self.model.predict(
            source=str(image_path),
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        # results is a list (one item per image); here it's length 1
        r = results[0]
        dets: List[Detection] = []

        if r.boxes is None or len(r.boxes) == 0:
            return dets

        xyxy = r.boxes.xyxy.cpu().numpy()   # (N,4)
        confs = r.boxes.conf.cpu().numpy()  # (N,)
        clss = r.boxes.cls.cpu().numpy()    # (N,)

        for i in range(xyxy.shape[0]):
            dets.append(
                Detection(
                    bbox_xyxy=xyxy[i].tolist(),
                    conf=float(confs[i]),
                    cls=int(clss[i]),
                )
            )

        return dets

    def predict_many(self, image_paths: List[Union[str, Path]]) -> List[List[Detection]]:
        """
        Batch predict (still fine if you pass many images).
        Returns list-of-detections per image, aligned with input order.
        """
        # Ultralytics can take a list of paths
        results = self.model.predict(
            source=[str(p) for p in image_paths],
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        all_out: List[List[Detection]] = []
        for r in results:
            dets: List[Detection] = []
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy()
                for i in range(xyxy.shape[0]):
                    dets.append(
                        Detection(
                            bbox_xyxy=xyxy[i].tolist(),
                            conf=float(confs[i]),
                            cls=int(clss[i]),
                        )
                    )
            all_out.append(dets)

        return all_out


if __name__ == "__main__":
    weights = r"C:\Windows\System32\runs\detect\train\weights\best.pt"

    detector = StepPanelDetector(weights_path=weights, conf=0.25, iou=0.7, device="cuda:0")

    dets = detector.predict_image(r"C:\Users\Kenan\Desktop\IKEA_Project\IKEA_assembly_guide_assistent\images\image_side2.jpg")
    print("detections:", len(dets))
    if dets:
        print(dets[0])
