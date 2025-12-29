import os
from pymongo import MongoClient
import gridfs
from bson import ObjectId

class IKEADatabase:
    def __init__(self, uri="mongodb://localhost:27017/", db_name="ikea_database"):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.fs = gridfs.GridFS(self.db)
        self.files_collection = self.db["fs.files"]

    def get_all_products(self):
        """
        Returns a list of unique categories and product names found in the database.
        """
        pipeline = [
            {
                "$group": {
                    "_id": {
                        "category": "$metadata.category",
                        "product_name": "$metadata.product_name"
                    }
                }
            },
            {"$sort": {"_id.category": 1, "_id.product_name": 1}}
        ]
        results = list(self.files_collection.aggregate(pipeline))
        
        products = []
        for r in results:
            if r["_id"].get("product_name"):
                products.append({
                    "category": r["_id"].get("category", "Uncategorized"),
                    "product_name": r["_id"].get("product_name")
                })
        return products

    def get_manual_text_by_product(self, product_name):
        """
        Retrieves the translated text instructions for a specific product.
        """
        doc = self.files_collection.find_one({"metadata.product_name": product_name})
        if doc and "metadata" in doc and "instructions_text" in doc["metadata"]:
            return doc["metadata"]["instructions_text"]
        return None

    def get_manual_text_by_filename(self, filename):
        """
        Retrieves text by filename (useful for checking uploads).
        """
        doc = self.files_collection.find_one({"filename": filename})
        if doc and "metadata" in doc and "instructions_text" in doc["metadata"]:
            return doc["metadata"]["instructions_text"]
        return None

    def save_manual_text(self, filename, text):
        """
        Updates the file document with the generated text instructions.
        """
        self.files_collection.update_one(
            {"filename": filename},
            {"$set": {"metadata.instructions_text": text}}
        )

    def upload_file(self, file_path, filename, category="Uncategorized", product_name=None):
        """
        Uploads a file to GridFS if it doesn't exist.
        """
        if not product_name:
            product_name = filename.replace(".pdf", "").replace("_", " ").title()

        # Check existing
        existing = self.files_collection.find_one({"filename": filename})
        if existing:
            return existing["_id"]

        with open(file_path, 'rb') as f:
            file_id = self.fs.put(
                f,
                filename=filename,
                metadata={
                    "category": category,
                    "product_name": product_name,
                    "source": "web_upload",
                    "instructions_text": None # Initialize empty
                }
            )
        return file_id

# --- For backward compatibility / standalone ingestion ---
if __name__ == "__main__":
    # Example standalone usage
    BASE_PATH = "/Users/andreasblock/Desktop/5. Semester/Systems and Software Engineering/Task/pdfs"
    db = IKEADatabase()
    
    if os.path.exists(BASE_PATH):
        print(f"📂 Scanning: {BASE_PATH}...\n")
        for dirpath, dirnames, filenames in os.walk(BASE_PATH):
            for filename in filenames:
                if filename.endswith(".pdf"):
                    full_path = os.path.join(dirpath, filename)
                    relative_path = os.path.relpath(full_path, BASE_PATH)
                    path_parts = relative_path.split(os.sep)
                    
                    if len(path_parts) >= 3:
                        category = path_parts[0]
                        product_name = path_parts[1].replace("_", " ").title()
                        
                        db.upload_file(full_path, filename, category, product_name)
                        print(f"✅ Processed: {product_name}")