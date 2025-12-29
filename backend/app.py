import os
import shutil
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# AI Logic
from backend.ai.models import use_agents
# Database Logic
from backend.database.mongodb import IKEADatabase

app = FastAPI(title="IKEA Assembly Assistant API")

# --- CONFIGURATION ---
UPLOAD_DIR = Path("uploads")
ARTIFACTS_DIR = Path("artifacts")
WEIGHTS_PATH = Path("weights/yolo_weights.pt")

UPLOAD_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Initialize DB
ikea_db = IKEADatabase()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "null"], # "null" allows opening html file directly
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=ARTIFACTS_DIR), name="static")

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
        # 1. Check DB for existing text by filename
        existing_text = ikea_db.get_manual_text_by_filename(file.filename)
        if existing_text:
            return {
                "status": "success", 
                "filename": file.filename, 
                "instructions": existing_text,
                "source": "database_cache"
            }

        # 2. Upload to GridFS (if not already there)
        # We infer product name from filename for new uploads
        product_name = file.filename.replace(".pdf", "").replace("_", " ").title()
        ikea_db.upload_file(str(file_location), file.filename, category="User Uploaded", product_name=product_name)

        # 3. Run AI Pipeline
        graph, final_state = use_agents(
            path_pdfnode=str(ARTIFACTS_DIR),
            path_detectornode=str(WEIGHTS_PATH), 
            path_cropnode=str(ARTIFACTS_DIR / "crops"),
            path_to_pdf=str(file_location)
        )
        
        final_output = final_state.get("final_output", "Processing complete, but no text generated.")

        # 4. Save generated text to DB
        ikea_db.save_manual_text(file.filename, final_output)
        
        return {
            "status": "success",
            "filename": file.filename,
            "instructions": final_output,
            "source": "ai_pipeline"
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "running"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)