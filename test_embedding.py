import os
from dotenv import dotenv
from google import genai
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
text = "How did i reset MFA after losing my phine?"
response = genai.embed_content(model = "models/embedding-001",
content = text,
task_type = "retrieval_query")

embedding = response["embedding"]
print("Embedding length: " , len(embedding))
print("First 10 values: ", embedding[:10])