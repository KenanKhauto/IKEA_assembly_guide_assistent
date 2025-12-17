from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import os
import PIL.Image
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches


@dataclass
class YoloSample:
    image_path: Path
    label_path: Path
    image: PIL.Image.Image
    boxes_xyxy: torch.Tensor      # (N, 4) in pixels: [x1, y1, x2, y2]
    class_ids: torch.Tensor       # (N,) int64
    size_hw: Tuple[int, int]      # (H, W)


class YoloFolderDataset(Dataset):
    """
    Dataset layout (flat):
      dataset/
        images/
          *.png (or jpg)
        labels/
          *.txt  (YOLO format)
        classes.txt (optional)

    YOLO label format per line:
      <class_id> <x_center> <y_center> <w> <h>
    All normalized to [0,1] relative to image size.
    """

    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        classes_path: str | Path | None = None,
        image_exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp"),
        return_pil: bool = True,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.return_pil = return_pil

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images dir not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels dir not found: {self.labels_dir}")

        self.image_paths = sorted(
            [p for p in self.images_dir.iterdir() if p.suffix.lower() in image_exts]
        )
        if not self.image_paths:
            raise RuntimeError(f"No images found in: {self.images_dir}")

        self.class_names = None
        if classes_path is not None:
            cp = Path(classes_path)
            if cp.exists():
                self.class_names = [line.strip() for line in cp.read_text(encoding="utf-8").splitlines() if line.strip()]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.image_paths[idx]
        label_path = self.labels_dir / (img_path.stem + ".txt")

        img = PIL.Image.open(img_path).convert("RGB")
        w, h = img.size

        class_ids, boxes_xyxy = self._load_yolo_labels(label_path, w, h)

        sample = {
            "image_path": str(img_path),
            "label_path": str(label_path),
            "width": w,
            "height": h,
            "boxes_xyxy": boxes_xyxy,     # Tensor (N,4) float32 pixels
            "class_ids": class_ids,       # Tensor (N,) int64
        }

        if self.return_pil:
            sample["image"] = img
        else:
            # If you want torch tensors: convert to CHW float in [0,1]
            img_t = torch.from_numpy(__import__("numpy").array(img)).permute(2, 0, 1).float() / 255.0
            sample["image"] = img_t

        return sample

    def _load_yolo_labels(self, label_path: Path, img_w: int, img_h: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          class_ids: (N,) int64
          boxes_xyxy: (N,4) float32 in pixels [x1,y1,x2,y2]
        """
        if not label_path.exists():
            # negative example
            return torch.zeros((0,), dtype=torch.int64), torch.zeros((0, 4), dtype=torch.float32)

        lines = label_path.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            return torch.zeros((0,), dtype=torch.int64), torch.zeros((0, 4), dtype=torch.float32)

        cls_list = []
        box_list = []

        for ln in lines:
            parts = ln.strip().split()
            if len(parts) != 5:
                # skip malformed line
                continue

            cls = int(float(parts[0]))
            xc = float(parts[1])
            yc = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])

            # normalized -> pixel
            x1 = (xc - bw / 2.0) * img_w
            y1 = (yc - bh / 2.0) * img_h
            x2 = (xc + bw / 2.0) * img_w
            y2 = (yc + bh / 2.0) * img_h

            # clamp
            x1 = max(0.0, min(x1, img_w - 1.0))
            y1 = max(0.0, min(y1, img_h - 1.0))
            x2 = max(0.0, min(x2, img_w - 1.0))
            y2 = max(0.0, min(y2, img_h - 1.0))

            cls_list.append(cls)
            box_list.append([x1, y1, x2, y2])

        if not box_list:
            return torch.zeros((0,), dtype=torch.int64), torch.zeros((0, 4), dtype=torch.float32)

        class_ids = torch.tensor(cls_list, dtype=torch.int64)
        boxes_xyxy = torch.tensor(box_list, dtype=torch.float32)
        return class_ids, boxes_xyxy


def show_sample(sample: Dict[str, Any], class_names: Optional[List[str]] = None, max_boxes: int = 200) -> None:
    """
    Visualize one dataset sample: image + rectangles.
    Works with sample["image"] as PIL image or torch tensor.
    """
    img = sample["image"]
    if isinstance(img, torch.Tensor):
        # CHW -> HWC for plotting
        img_np = img.permute(1, 2, 0).cpu().numpy()
    else:
        img_np = img

    boxes = sample["boxes_xyxy"]
    class_ids = sample["class_ids"]

    fig, ax = plt.subplots()
    ax.imshow(img_np)

    n = min(len(boxes), max_boxes)
    for i in range(n):
        x1, y1, x2, y2 = boxes[i].tolist()
        w = x2 - x1
        h = y2 - y1

        rect = patches.Rectangle((x1, y1), w, h, fill=False, linewidth=2)
        ax.add_patch(rect)

        if len(class_ids) > i:
            cls_id = int(class_ids[i].item())
            label = str(cls_id)
            if class_names and 0 <= cls_id < len(class_names):
                label = class_names[cls_id]
            ax.text(x1, max(0, y1 - 5), label, fontsize=10, va="bottom")

    ax.set_title(Path(sample["image_path"]).name)
    ax.axis("off")
    plt.show()


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    dataset_root = Path(r"C:\Users\Kenan\Desktop\IKEA_Project\IKEA_assembly_guide_assistent\dataset")
    ds = YoloFolderDataset(
        images_dir=dataset_root / "images",
        labels_dir=dataset_root / "labels",
        classes_path=dataset_root / "classes.txt",  # optional
        return_pil=True,
    )

    # show a few samples
    for i in [3]:
        sample = ds[i]
        show_sample(sample, class_names=ds.class_names)
