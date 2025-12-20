from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union


# -----------------------------
# State / inputs
# -----------------------------

class PdfSource(TypedDict, total=False):
    kind: Literal["path", "bytes"]
    path: str
    bytes: bytes
    filename: str  # optional, useful for logging/UI


class PdfToImagesConfig(TypedDict, total=False):
    artifact_root: str                 # base directory for artifacts
    cache_policy: Literal["use_cache", "refresh"]
    dpi: int                           # default 200
    format: Literal["png"]             # keep "png" for now
    renderer_version: str              # cache-busting if renderer changes


# -----------------------------
# Node implementation
# -----------------------------

@dataclass(frozen=True)
class PdfToImagesNode:
    artifact_root: Path = Path("./artifacts")
    cache_policy: Literal["use_cache", "refresh"] = "use_cache"
    dpi: int = 200
    image_format: Literal["png"] = "png"
    renderer_version: str = "pymupdf_v1"

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        LangGraph node: reads state["pdf_source"], renders all pages,
        stores images in cached artifact store, returns state update under "pdf_render".
        """
        pdf_source = state.get("pdf_source")
        if not isinstance(pdf_source, dict):
            raise ValueError("Expected state['pdf_source'] to be a dict with kind/path/bytes.")

        # Allow per-run overrides via state["pdf_to_images_config"] (optional)
        cfg = state.get("pdf_to_images_config") or {}
        artifact_root = Path(cfg.get("artifact_root", str(self.artifact_root)))
        cache_policy = cfg.get("cache_policy", self.cache_policy)
        dpi = int(cfg.get("dpi", self.dpi))
        image_format = cfg.get("format", self.image_format)
        renderer_version = cfg.get("renderer_version", self.renderer_version)

        if image_format != "png":
            raise ValueError("This node currently supports only PNG output (format='png').")

        pdf_bytes = self._load_pdf_bytes(pdf_source)
        pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()

        cache_key = self._make_cache_key(
            pdf_hash=pdf_hash,
            dpi=dpi,
            image_format=image_format,
            renderer_version=renderer_version,
        )
        out_dir = artifact_root / "pdf_pages" / cache_key
        manifest_path = out_dir / "manifest.json"

        # Cache hit path
        if cache_policy == "use_cache" and manifest_path.exists():
            manifest = self._read_json(manifest_path)
            return {"pdf_render": manifest}

        # Refresh path: clear directory
        out_dir.mkdir(parents=True, exist_ok=True)
        if cache_policy == "refresh":
            # remove old pngs + manifest if present
            for p in out_dir.glob("page_*.png"):
                try:
                    p.unlink()
                except OSError:
                    pass
            if manifest_path.exists():
                try:
                    manifest_path.unlink()
                except OSError:
                    pass

        # Render
        pages = self._render_all_pages_pymupdf(pdf_bytes, out_dir, dpi=dpi)

        manifest = {
            "cache_key": cache_key,
            "pdf_hash": pdf_hash,
            "renderer_version": renderer_version,
            "page_count": len(pages),
            "dpi": dpi,
            "format": image_format,
            "pages": pages,  # 0-based indices
        }
        title_page = None
        if pages and isinstance(pages[0], dict) and "image_path" in pages[0]:
            title_page = {
                "page_index": int(pages[0].get("page_index", 0)),
                "image_path": pages[0]["image_path"],
                "width_px": pages[0].get("width_px"),
                "height_px": pages[0].get("height_px"),
                "dpi": dpi,
            }

        self._write_json(manifest_path, manifest)
        return {
                    "pdf_render": manifest,
                    "manual_id": pdf_hash,
                    "title_page": title_page,
                }

    # -----------------------------
    # Helpers
    # -----------------------------

    def _load_pdf_bytes(self, pdf_source: PdfSource) -> bytes:
        kind = pdf_source.get("kind")
        if kind == "path":
            path = pdf_source.get("path")
            if not path:
                raise ValueError("pdf_source.kind='path' requires pdf_source.path")
            with open(path, "rb") as f:
                return f.read()

        if kind == "bytes":
            b = pdf_source.get("bytes")
            if b is None:
                raise ValueError("pdf_source.kind='bytes' requires pdf_source.bytes")
            return b

        raise ValueError("pdf_source.kind must be 'path' or 'bytes'")

    def _render_all_pages_pymupdf(self, pdf_bytes: bytes, out_dir: Path, dpi: int) -> List[Dict[str, Any]]:
        """
        Renders all pages using PyMuPDF (fitz).
        Produces page_000.png, page_001.png, ... and returns metadata records.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError as e:
            raise ImportError(
                "PyMuPDF is required for rendering PDFs in this node. Install with: pip install pymupdf"
            ) from e

        # PyMuPDF uses 72 DPI as baseline scale
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages: List[Dict[str, Any]] = []

        for i in range(doc.page_count):
            try:
                page = doc.load_page(i)
                pix = page.get_pixmap(matrix=matrix, alpha=False)  # alpha False = smaller PNG
                filename = f"page_{i:03d}.png"
                image_path = out_dir / filename
                pix.save(str(image_path))

                pages.append({
                    "page_index": i,                 # 0-based
                    "image_path": str(image_path),   # local path reference
                    "width_px": pix.width,
                    "height_px": pix.height,
                    "dpi": dpi,
                })
            except Exception as ex:
                # Keep going; downstream can decide what to do with errors
                pages.append({
                    "page_index": i,
                    "error": f"{type(ex).__name__}: {ex}",
                    "dpi": dpi,
                })

        doc.close()
        return pages

    def _make_cache_key(self, pdf_hash: str, dpi: int, image_format: str, renderer_version: str) -> str:
        raw = f"{pdf_hash}_dpi{dpi}_{image_format}_{renderer_version}"
        return self._sanitize_path_component(raw)

    def _sanitize_path_component(self, s: str) -> str:
        """
        Make a string safe to use as a single directory/file name component on Windows/macOS/Linux.
        """
        # Windows forbids <>:"/\\|?* (and some other edge cases). Keep it simple and portable:
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
        sanitized = sanitized.strip(" ._")
        return sanitized or "cache"

    def _read_json(self, path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_json(self, path: Path, obj: Dict[str, Any]) -> None:
        tmp = Path(str(path) + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)


# -----------------------------
# Example usage (local PDFs now)
# -----------------------------
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    node = PdfToImagesNode(artifact_root=Path("./artifacts"), dpi=200)

    state = {
        "pdf_source": {
            "kind": "path",
            "path": BASE_DIR / ".." / ".." / "pdfs" / "ikea2.pdf",
            "filename": "ikea.pdf",
        },
        # Optional overrides:
        # "pdf_to_images_config": {"dpi": 200, "cache_policy": "use_cache"}
    }

    out = node(state)
    print(out["pdf_render"]["page_count"])
    print(out["pdf_render"]["pages"][0])
