import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "support_knowledge"
TOP_K = 3
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not est in .env file")

gemini_client = genai.Client(api_key = GEMINI_API_KEY)
qdrant_client = QdrantClient(path = "qdrant_data")

query = "How did i reset MFA after losing my phone?"
embedding_response = gemini_client.models.embed_content(
    model = "gemini-embedding-001",
    contents = [query],
    config = types.EmbedContentConfig(
        task_type = "RETRIEVAL_QUERY"
    )
)

query_vector = embedding_response.embeddings[0].values
print(f"Query: {query}")
print(f"Query embedding dimesnion: {len(query_vector)}")

results = qdrant_client.query_points(collection_name=COLLECTION_NAME,
                                     query = query_vector,
                                     limit = TOP_K,
                                     with_payload=True)

points = results.points
if not points:
    print("\n No matching chunks found")
else:
    print(f"\n Top {len(points)} matches")
    for i, point in enumerate(points, start=1):
        payload = point.payload or {}
        chunk_text = (
            payload.get("text") or
            payload.get("chunk") or
            payload.get("content") or
            "[No chunk found in payload]"
        )
        source = payload.get("source", "Unknwon source")
        chunk_id = payload.get("chunk_id", "Unknown chunk id")
        print(f"result {i}")
        print(f"Score {point.score}")
        print(f"source {source}")
        print(f"chunk id {chunk_id}")
        print(f"text {chunk_text}")
        print("-" * 80)
        
qdrant_client.close()




