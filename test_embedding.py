import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Create Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

text = "How did I reset MFA after losing my phone?"

# Generate embedding
response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=[text],
    config=types.EmbedContentConfig(
        task_type="RETRIEVAL_QUERY"
    ),
)

embedding = response.embeddings[0].values

print("Embedding length:", len(embedding))
print("First 10 values:", embedding[:10])