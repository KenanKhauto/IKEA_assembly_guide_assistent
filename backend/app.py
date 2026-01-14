import os
import shutil
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# AI Logic
from .ai.models import use_agents
# Database Logic
from .database.mongodb import IKEADatabase
import json
import re

def extract_dict_from_raw(raw: str) -> dict:
    if not raw:
        raise ValueError("Empty raw instructions")

    raw = raw.strip()

    # Handle ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
    if match:
        raw = match.group(1)

    return json.loads(raw)


app = FastAPI(title="IKEA Assembly Assistant API")

# --- CONFIGURATION ---
UPLOAD_DIR = Path("uploads")
ARTIFACTS_DIR = Path("artifacts")
WEIGHTS_PATH = Path("weights/best.pt")

UPLOAD_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Initialize DB
ikea_db = IKEADatabase("mongodb+srv://kenan:Yyecgaa123123@cluster0.s0aykgz.mongodb.net/?", "ikea_database")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "null"], # "null" allows opening html file directly
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=ARTIFACTS_DIR), name="static")


@app.get("/test")
def test_backend():
    return {"Message": "I am working with my nose"}

@app.get("/products")
def get_products():
    """Returns a list of all products in the database for the dropdown."""
    try:
        return ikea_db.get_all_products()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/manual-text")
def get_manual_text(product_name: str = Query(..., description="The name of the product")):
    """Returns the cached text instructions for a specific product."""
    text = ikea_db.get_manual_text_by_product(product_name)
    if text:
        return {"status": "success", "product_name": product_name, "instructions": text, "source": "database"}
    else:
        raise HTTPException(status_code=404, detail="Instructions not found for this product.")

@app.post("/process-manual")
async def process_manual_endpoint(file: UploadFile = File(...)):
    """
    1. Uploads file to DB if new.
    2. Checks if text exists in DB.
    3. If not, runs AI, saves text to DB, and returns it.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    file_location = UPLOAD_DIR / file.filename
    
    # Save locally temporarily for processing
    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file locally: {str(e)}")

    
    try:

        # --- 1) Upload (or reuse) + compute hash ---
        file_id, content_hash, existed_before = ikea_db.upload_file_cached(
            file_path=str(file_location),
            original_filename=file.filename,
            category="web_upload",   # or pass from UI later
            product_name=None
        )

        # --- 2) Cache hit? return stored analysis / instructions ---
        cached_analysis = ikea_db.get_analysis_by_hash(content_hash)
        if cached_analysis:
            # If you also store plain text, prefer that:
            steps = cached_analysis.get("assembly_instructions", [])
            return {
                "status": "success",
                "filename": file.filename,
                "content_hash": content_hash,
                "cached": True,
                "assembly_instructions": steps,
                # "analysis": cached_analysis,
                "source": "database_cache",
                "cache_key": cached_analysis.get("cache_key"),
            }

        # 3. Run AI Pipeline
        graph, final_state = use_agents(
            path_pdfnode=str(ARTIFACTS_DIR),
            path_detectornode=str(WEIGHTS_PATH), 
            path_cropnode=str(ARTIFACTS_DIR / "crops"),
            path_to_pdf=str(file_location)
        )
        
        final_output = final_state.get("final_instructions", "Processing complete, but no text generated.")
        cache_key = final_state.get("pdf_render", {}).get("cache_key")
        # unwrap {"raw": "..."} if needed
        if isinstance(final_output, dict):
            raw_text = final_output.get("raw", "")
        else:
            raw_text = final_output

        assembly_dict = extract_dict_from_raw(raw_text)

        # 4. Save generated text to DB
        ikea_db.save_analysis(content_hash, {
            "final_instructions": final_output,
            # "final_state": final_state,   # optional: can be big; remove if too large
            "pipeline": "use_agents_v1",
            "assembly_instructions": assembly_dict["assembly_instructions"],
            "cache_key":cache_key
        })
        # optional: store plain text separately too
        # ikea_db.save_instructions_text(content_hash, final_output)

        return {
            "status": "success",
            "filename": file.filename,
            "content_hash": content_hash,
            "cache_key":cache_key,
            "cached": False,
            "assembly_instructions": assembly_dict["assembly_instructions"],
            "source": "ai_pipeline",
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        # optional cleanup (prevents uploads folder from growing forever)
        try:
            if file_location.exists():
                file_location.unlink()
        except Exception:
            pass


@app.get("/health")
def health_check():
    return {"status": "running"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)