import os
from pymongo import MongoClient
import gridfs

# --- CONFIGURATION ---
# 1. Your specific local path
BASE_PATH = "/Users/andreasblock/Desktop/5. Semester/Systems and Software Engineering/Task/pdfs"

# 2. Database Connection
# Update the string if your DB is not local (e.g., MongoDB Atlas)
client = MongoClient("mongodb://localhost:27017/")
db = client["ikea_database"]
fs = gridfs.GridFS(db)

def ingest_manuals(root_folder):
    if not os.path.exists(root_folder):
        print(f"❌ Error: The path '{root_folder}' does not exist.")
        return

    print(f"📂 Scanning: {root_folder}...\n")
    
    # Walk through the directory tree
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(".pdf"):
                full_path = os.path.join(dirpath, filename)
                
                # --- METADATA EXTRACTION ---
                # We calculate the relative path to figure out Category and Product
                # Example: /.../pdfs/Chair/applaro/0.pdf
                # relative_path = "Chair/applaro/0.pdf"
                relative_path = os.path.relpath(full_path, root_folder)
                path_parts = relative_path.split(os.sep)
                
                # Based on your tree structure:
                # Part 0 = Category (e.g., "Chair")
                # Part 1 = Product Name (e.g., "applaro")
                # Part 2 = Filename (e.g., "0.pdf")
                
                if len(path_parts) >= 3:
                    category = path_parts[0]
                    product_name = path_parts[1].replace("_", " ").title() # Clean up name
                    
                    # Check if file already exists in DB to avoid duplicates
                    existing = db.fs.files.find_one({
                        "metadata.category": category, 
                        "metadata.product_name": product_name,
                        "filename": filename
                    })

                    if existing:
                        print(f"⚠️  Skipping duplicate: {category} - {product_name}")
                        continue

                    # --- UPLOAD TO MONGODB ---
                    with open(full_path, 'rb') as f:
                        fs.put(
                            f,
                            filename=filename,
                            metadata={
                                "category": category,
                                "product_name": product_name,
                                "original_path": relative_path,
                                "source": "local_disk"
                            }
                        )
                    print(f"✅ Uploaded: [{category}] {product_name} ({filename})")

if __name__ == "__main__":
    ingest_manuals(BASE_PATH)