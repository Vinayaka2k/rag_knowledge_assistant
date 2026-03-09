import os
from typing import List
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types
from qdrant_client import QdrantClient

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COLLECTION_NAME = "support_knowledge"
EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-2.5-flash"
TOP_K = 3

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not present in your .env file")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
qdrant_client = QdrantClient(
    host="qdrant",
    port=6333
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    qdrant_client.close()


app = FastAPI(
    title="Support Knowledge Copilot",
    lifespan=lifespan
)


class AskRequest(BaseModel):
    question: str


class SourceItem(BaseModel):
    source: str
    chunk_id: str
    score: float
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceItem]


def embed_query(query: str) -> List[float]:
    response = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[query],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    return response.embeddings[0].values


def retrieve_chunks(query_vector: List[float], top_k: int = TOP_K):
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True
    )
    return results.points


def build_context(points) -> str:
    context_parts = []

    for idx, point in enumerate(points, start=1):
        payload = point.payload or {}

        chunk_text = (
            payload.get("text")
            or payload.get("chunk")
            or payload.get("content")
            or ""
        )

        source = payload.get("source", "unknown_source")
        chunk_id = payload.get("chunk_id", f"chunk_{idx}")

        if chunk_text.strip():
            context_parts.append(
                f"[Source: {source} | Chunk ID: {chunk_id}]\n{chunk_text}"
            )

    return "\n\n---\n\n".join(context_parts)


def generate_answer(question: str, context: str) -> str:
    prompt = f"""
You are a helpful enterprise IT support assistant.
Answer the user's question using ONLY the provided context.
If the answer is not present in the context, say:
"I could not find the answer in the retrieved documents."

Be clear, accurate, and concise.

Context:
{context}

User question:
{question}
"""

    response = gemini_client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt
    )
    return response.text.strip()


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Support Knowledge Copilot API is running"
    }


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        query_vector = embed_query(question)
        points = retrieve_chunks(query_vector, TOP_K)

        if not points:
            return AskResponse(
                question=question,
                answer="I couldn't find any relevant documents.",
                sources=[]
            )

        context = build_context(points)
        answer = generate_answer(question, context)

        sources = []
        for idx, point in enumerate(points, start=1):
            payload = point.payload or {}

            chunk_text = (
                payload.get("text")
                or payload.get("chunk")
                or payload.get("content")
                or ""
            )

            sources.append(
                SourceItem(
                    source=payload.get("source", "unknown_source"),
                    chunk_id=payload.get("chunk_id", f"chunk_{idx}"),
                    score=float(point.score),
                    text=chunk_text
                )
            )

        return AskResponse(
            question=question,
            answer=answer,
            sources=sources
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))