from __future__ import annotations
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List

import google.generativeai as genai
from qdrant_client import QDrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from dotenv import load_dotenv
load_dotenv()
CHUNKS_FILE = Path("/data/processed/chunks.json")
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "support_knowledge"
EMBEDDING_MODEL = "models/embedding-001"
def load_chunks(file_path: Path) -> List[Dict[str, Any]]:
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def configure_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found")
    genai.configure(api_key=api_key)

def embed_text(text: str) -> List[float]:
    result = genai.embed_content(
        model = EMBEDDING_MODEL,
        content = text,
        task_type = "retrieval document"
    )
    return result["embedding"]

def recreate_collection(qdrant: QDrantClient, vector_size: int):
    collections = qdrant.get_collections().collections()
    names = [c.name for c in collections]
    if COLLECTION_NAME in names:
        qdrant.delete_collection(collection_name = COLLECTION_NAME)
    qdrant.create_collection(
        collection_name = COLLECTION_NAME,
        vectors_config = VectorParams(size = vector_size, distance = Distance.COSINE)
    )

def main():
    print("Loading chunks...")
    chunks = load_chunks(CHUNKS_FILE)
    configure_gemini()
    print("Generating Embeddings...")
    embeddings = []
    for chunk in chunks:
        emb = embed_text(chunk["text"])
        embeddings.append(emb)
    vector_size = len(embeddings[0])
    print("Embedding dimension: " + vector_size)

    print("Connectig to Qdrant...")
    qdrant = QDrantClient(host = QDRANT_HOST, port = QDRANT_PORT)

    print("Creating collection...")
    recreate_collection(qdrant, vector_size)

    points = []

    for chunk, emb in zip(chunks, embeddings):
        point = PointStruct(
            id = str(uuid.uuid4()),
            vector = emb,
            payload = {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"]
            }
        )
        points.append(point)
    print("Uploading vectors...")
    qdrant.upsert(collection_name = COLLECTION_NAME,
                  points = points)
    
    print("Done. strored ", len(points), " chunks in Qdrant.")


if __name__ == "__main__":
    main()











