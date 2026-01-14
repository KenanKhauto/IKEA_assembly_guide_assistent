import os
import hashlib
from datetime import datetime
from pymongo import MongoClient
import gridfs

class IKEADatabase:
    def __init__(self, uri=None, db_name=None):
        # Prefer env vars
        uri = uri or os.getenv("MONGO_URI")
        db_name = db_name or os.getenv("MONGO_DB_NAME", "ikea_database")

        if not uri:
            raise ValueError("MONGO_URI not set. Provide uri=... or set env var MONGO_URI.")

        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.fs = gridfs.GridFS(self.db)
        self.files_collection = self.db["fs.files"]

        # Optional but recommended indexes:
        # - content_hash lookup fast
        # - product dropdown fast
        self.files_collection.create_index("metadata.content_hash")
        self.files_collection.create_index("metadata.product_name")
        self.files_collection.create_index("metadata.category")

    # ---------- helpers ----------
    @staticmethod
    def sha256_of_file(file_path: str) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def find_by_hash(self, content_hash: str):
        return self.files_collection.find_one({"metadata.content_hash": content_hash})

    # ---------- queries ----------
    def get_all_products(self):
        pipeline = [
            {"$match": {"metadata.product_name": {"$ne": None}}},
            {
                "$group": {
                    "_id": {"category": "$metadata.category", "product_name": "$metadata.product_name"}
                }
            },
            {"$sort": {"_id.category": 1, "_id.product_name": 1}}
        ]
        results = list(self.files_collection.aggregate(pipeline))
        return [
            {
                "category": r["_id"].get("category", "Uncategorized"),
                "product_name": r["_id"].get("product_name"),
            }
            for r in results
            if r["_id"].get("product_name")
        ]

    def get_manual_text_by_product(self, product_name: str):
        doc = self.files_collection.find_one({"metadata.product_name": product_name})
        return (doc or {}).get("metadata", {}).get("instructions_text")

    def get_analysis_by_hash(self, content_hash: str):
        doc = self.find_by_hash(content_hash)
        if not doc:
            return None
        return doc.get("metadata", {}).get("analysis")

    # ---------- writes ----------
    def upload_file_cached(
        self,
        file_path: str,
        original_filename: str,
        category: str = "Uncategorized",
        product_name: str | None = None,
        source: str = "web_upload",
        store_local_path: bool = False,
    ):
        """
        Uploads PDF to GridFS only if content_hash doesn't exist.
        Returns (file_id, content_hash, existed_before)
        """
        if not product_name:
            product_name = original_filename.replace(".pdf", "").replace("_", " ").title()

        content_hash = self.sha256_of_file(file_path)

        existing = self.find_by_hash(content_hash)
        if existing:
            return existing["_id"], content_hash, True

        # Make GridFS filename unique to avoid collisions
        stored_filename = f"{content_hash}_{original_filename}"

        meta = {
            "category": category,
            "product_name": product_name,
            "source": source,
            "content_hash": content_hash,
            "original_filename": original_filename,
            "stored_filename": stored_filename,
            "instructions_text": None,
            "analysis": None,
            "analysis_version": 1,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }

        if store_local_path:
            meta["local_path"] = file_path

        with open(file_path, "rb") as f:
            file_id = self.fs.put(f, filename=stored_filename, metadata=meta)

        return file_id, content_hash, False

    def save_analysis(self, content_hash: str, analysis: dict):
        self.files_collection.update_one(
            {"metadata.content_hash": content_hash},
            {"$set": {"metadata.analysis": analysis, "metadata.updated_at": datetime.utcnow()}}
        )

    def save_instructions_text(self, content_hash: str, text: str):
        self.files_collection.update_one(
            {"metadata.content_hash": content_hash},
            {"$set": {"metadata.instructions_text": text, "metadata.updated_at": datetime.utcnow()}}
        )


if __name__ == "__main__":
    db = IKEADatabase("mongodb+srv://kenan:Yyecgaa123123@cluster0.s0aykgz.mongodb.net/?")
    p = db.get_all_products()
    print(p)
    