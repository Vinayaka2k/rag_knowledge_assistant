from __future__ import annotations
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List

from google import genai
from google.genai import types

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from dotenv import load_dotenv

load_dotenv()

CHUNKS_FILE = Path("data/processed/chunks.json")
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "support_knowledge"
EMBEDDING_MODEL = "gemini-embedding-001"


def load_chunks(file_path: Path) -> List[Dict[str, Any]]:
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found")
    return genai.Client(api_key=api_key)


def embed_text(client, text: str) -> List[float]:
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT"
            # optional:
            # output_dimensionality=768
        )
    )
    return result.embeddings[0].values


def recreate_collection(qdrant: QdrantClient, vector_size: int):
    collections = qdrant.get_collections().collections
    names = [c.name for c in collections]

    if COLLECTION_NAME in names:
        qdrant.delete_collection(collection_name=COLLECTION_NAME)

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def main():
    print("Loading chunks...")
    chunks = load_chunks(CHUNKS_FILE)

    client = get_gemini_client()

    print("Generating embeddings...")
    embeddings = []
    for chunk in chunks:
        emb = embed_text(client, chunk["text"])
        embeddings.append(emb)

    vector_size = len(embeddings[0])
    print("Embedding dimension:", vector_size)

    print("Connecting to Qdrant...")
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    print("Creating collection...")
    recreate_collection(qdrant, vector_size)

    points = []

    for chunk, emb in zip(chunks, embeddings):
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=emb,
            payload={
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
            },
        )
        points.append(point)

    print("Uploading vectors...")
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print("Done. stored", len(points), "chunks in Qdrant.")


if __name__ == "__main__":
    main()