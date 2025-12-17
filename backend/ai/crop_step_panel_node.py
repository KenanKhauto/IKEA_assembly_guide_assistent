from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import os

from PIL import Image


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def _safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)


@dataclass
class CropStepPanelsNode:
    """
    Crops detected step panels into per-step images.

    Reads:
      state["pdf_render"]["cache_key"]
      state["pdf_render"]["pages"] -> list of {page_index, image_path, ...}
      state["panel_detections"]["by_page"] -> {page_index: [{bbox_xyxy, confidence, class_id}, ...]}

    Writes:
      state["step_crops"] = {
        "cache_key": ...,
        "steps_dir": ...,
        "step_count": ...,
        "steps": [
          {
            "step_id": ...,
            "page_index": ...,
            "panel_index": ...,
            "bbox_xyxy": [...],
            "confidence": ...,
            "image_path": ...,
            "width_px": ...,
            "height_px": ...,
          }, ...
        ]
      }
    """

    artifacts_root: Path = Path("./artifacts")
    padding_px: int = 16
    min_box_size_px: int = 10  # ignore tiny junk boxes
    cache_policy: str = "use_cache"  # "use_cache" | "refresh"
    steps_subdir: str = "step_crops"
    manifest_name: str = "manifest.json"

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pdf_render = state.get("pdf_render")
        if not pdf_render or "pages" not in pdf_render:
            raise ValueError("Missing state['pdf_render']['pages']")

        panel_det = state.get("panel_detections")
        if not panel_det or "by_page" not in panel_det:
            raise ValueError("Missing state['panel_detections']['by_page']")

        cache_key = pdf_render.get("cache_key")
        if not cache_key:
            raise ValueError("Missing state['pdf_render']['cache_key']")

        # Allow overrides
        cfg = state.get("crop_config") or {}
        padding_px = int(cfg.get("padding_px", self.padding_px))
        min_box_size_px = int(cfg.get("min_box_size_px", self.min_box_size_px))
        cache_policy = cfg.get("cache_policy", self.cache_policy)

        # Build page_index -> image_path map
        page_path: Dict[int, str] = {}
        for p in pdf_render["pages"]:
            if "image_path" in p and "page_index" in p:
                page_path[int(p["page_index"])] = p["image_path"]

        by_page: Dict[int, List[Dict[str, Any]]] = {
            int(k): v for k, v in panel_det["by_page"].items()
        }

        # Output folder
        out_dir = self.artifacts_root / self.steps_subdir / cache_key
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / self.manifest_name

        # Cache hit
        if cache_policy == "use_cache" and manifest_path.exists():
            return {"step_crops": self._read_json(manifest_path)}

        # Refresh
        if cache_policy == "refresh":
            for p in out_dir.glob("step_*.png"):
                try:
                    p.unlink()
                except OSError:
                    pass
            if manifest_path.exists():
                try:
                    manifest_path.unlink()
                except OSError:
                    pass

        pdf_hash_short = cache_key.split(":")[0][:6]

        steps: List[Dict[str, Any]] = []
        global_step_idx = 0

        # Process pages in order
        for page_index in sorted(by_page.keys()):
            img_path = page_path.get(page_index)
            if not img_path:
                # detection exists but image missing (render failed) -> skip
                continue

            dets = by_page.get(page_index, [])
            if not dets:
                continue

            # Open page image once
            img = Image.open(img_path).convert("RGB")
            W, H = img.size

            # Normalize/validate boxes + attach sorting keys
            cleaned: List[Tuple[float, float, float, float, Dict[str, Any]]] = []
            for d in dets:
                bbox = d.get("bbox_xyxy")
                if not bbox or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = map(float, bbox)

                # clamp
                x1 = _clamp(x1, 0.0, W - 1.0)
                x2 = _clamp(x2, 0.0, W - 1.0)
                y1 = _clamp(y1, 0.0, H - 1.0)
                y2 = _clamp(y2, 0.0, H - 1.0)

                # ensure correct ordering
                if x2 < x1:
                    x1, x2 = x2, x1
                if y2 < y1:
                    y1, y2 = y2, y1

                if (x2 - x1) < min_box_size_px or (y2 - y1) < min_box_size_px:
                    continue

                cleaned.append((x1, y1, x2, y2, d))

            if not cleaned:
                continue

            # Sort: top-to-bottom, left-to-right
            cleaned.sort(key=lambda t: (t[1], t[0]))  # (y1, x1)

            # Crop each panel
            for panel_index, (x1, y1, x2, y2, d) in enumerate(cleaned):
                # Add padding and clamp again
                px1 = int(_clamp(x1 - padding_px, 0, W - 1))
                py1 = int(_clamp(y1 - padding_px, 0, H - 1))
                px2 = int(_clamp(x2 + padding_px, 0, W - 1))
                py2 = int(_clamp(y2 + padding_px, 0, H - 1))

                # PIL crop expects (left, upper, right, lower) with right/lower exclusive-ish,
                # but it behaves well with ints; ensure at least 1px region.
                if px2 <= px1 or py2 <= py1:
                    continue

                crop = img.crop((px1, py1, px2, py2))

                # Stable step_id + filename
                step_id = f"{pdf_hash_short}_p{page_index:03d}_s{panel_index:02d}"
                fname = f"step_{step_id}.png"
                out_path = out_dir / fname
                crop.save(out_path)

                steps.append({
                    "step_id": step_id,
                    "global_step_index": global_step_idx,
                    "page_index": page_index,
                    "panel_index": panel_index,
                    "bbox_xyxy": [x1, y1, x2, y2],  # original bbox (no padding)
                    "bbox_xyxy_padded": [px1, py1, px2, py2],
                    "confidence": float(d.get("confidence", 0.0)),
                    "class_id": int(d.get("class_id", 0)),
                    "image_path": str(out_path),
                    "width_px": crop.size[0],
                    "height_px": crop.size[1],
                    "source_page_image_path": img_path,
                })
                global_step_idx += 1

        manifest = {
            "cache_key": cache_key,
            "steps_dir": str(out_dir),
            "step_count": len(steps),
            "padding_px": padding_px,
            "min_box_size_px": min_box_size_px,
            "steps": steps,
        }

        self._write_json(manifest_path, manifest)
        return {"step_crops": manifest}

    def _read_json(self, path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_json(self, path: Path, obj: Dict[str, Any]) -> None:
        tmp = Path(str(path) + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
