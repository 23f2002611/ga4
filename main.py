from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import google.generativeai as genai
import os

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

# Request schema
class SimilarityRequest(BaseModel):
    docs: List[str]
    query: str

# Cosine similarity
def cosine_similarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

@app.post("/similarity")
async def similarity_endpoint(req: SimilarityRequest):
    docs = req.docs
    query = req.query

    if not docs or not query:
        return {"error": "Both 'docs' and 'query' must be provided."}

    # Get embeddings for documents
    doc_embeddings = []
    for doc in docs:
        emb = genai.embed_content(
            model="models/embedding-001",  # Gemini embedding model
            content=doc
        )
        doc_embeddings.append(emb["embedding"])

    # Get embedding for query
    query_emb = genai.embed_content(
        model="models/embedding-001",
        content=query
    )["embedding"]

    # Compute cosine similarities
    similarities = [
        (doc, cosine_similarity(query_emb, emb))
        for doc, emb in zip(docs, doc_embeddings)
    ]

    # Sort by similarity score
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return top 3 matches
    top_matches = [doc for doc, _ in similarities[:3]]

    return {"matches": top_matches}
