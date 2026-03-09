import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
load_dotenv()

from typing import List, Dict, Any
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "support_knowledge"
EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-2.5-flash"
TOP_K = 3

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not present in your env file")

gemini_client = genai.Client(api_key = GEMINI_API_KEY)
qdrant_client = QdrantClient(path="qdrant_data")

def embed_query(query: str) -> List[float]:
    """
    Generate embedding for the user query
    """
    response = gemini_client.models.embed_content(
        model = EMBEDDING_MODEL,
        contents = [query],
        config = types.EmbedContentConfig(
            task_type = "RETRIEVAL_QUERY"
        )
    )
    return response.embeddings[0].values

def retrieve_chunks(query_vector: list[float], top_k: int = TOP_K):
    """
    Search qdrant and return the top K matching points
    """
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True
    )
    return results.points

def build_context(points) -> str:
    """
    Build context string from retrievd Qdrant points
    """
    context_parts = []
    for idx, point in enumerate(points, start=1):
        payload = point.payload or {}
        chunk_text = (
            payload.get("text") or
            payload.get("chunk") or
            payload.get("content") or
            ""
        )
        source = payload.get("source", "unknown source")
        chunk_id = payload.get("chunk_id", f"chunk_{idx}")
        if chunk_text.strip():
            context_parts.append(
                f"[Source: {source} | ChunkId: {chunk_id}] \n {chunk_text}"
            )
        return "\n\n --- \n\n".join(context_parts)
    

def generate_answer(query: str, context: str):
    """
    Genearte a final ansewr using the retrieved context
    """
    prompt = f"""
        You are helpful enterprise IT support assisant.
        Answer the usre's question using ONLY the context provided below.
        If the answer is not present in the context, 
        say: I couldnot find the asnwer i nthe retrieed documents.
        Keep the asnwer clean, acuracte and concise. 
        Context:
        {context}
        User quetsion:
        {query}
        """
    response = gemini_client.models.generate_content(
            model = GENERATION_MODEL,
            contents=prompt
        )
    return response.text

def main():
    query = "What if the user doesnt receive the email?"
    try:
        print(f"\nUser question: \n {query}")
        query_vector = embed_query(query)
        print(f"\n Query embedding dimension: {len(query_vector)}")

        points = retrieve_chunks(query_vector, TOP_K)

        if not points:
            print("\n No relevant chunks found in Qdrant")
            return
        print(f"\n Top {len(points)} retrieved chinks:\n")
        for i, point in enumerate(points, start=1):
            payload = point.payload or {}
            # print(f"Result: {i}")
            # print(f"Score:  {point.score}")
            # print(f"Source:  {payload.get('source', 'unknown_sourec')}")
            # print(f"Chunk id: , {payload.get('chunk_id', 'unknown chink')}")
            # print(f"Text :  {payload.get('text') or payload.get('chunk') or payload.get('content') or ''}")
            # print("-"*80)

        context = build_context(points)
        print("Context: ", context)
        final_answer = generate_answer(query, context)

        print("\n Final answer \n")
        print(final_answer)

    finally:
        qdrant_client.close()

if __name__ == "__main__":
    main()






