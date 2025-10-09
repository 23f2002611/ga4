from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from openai import OpenAI
import numpy as np

# Initialize OpenAI client
client = OpenAI()

# FastAPI app
app = FastAPI()

# Enable CORS (allow all origins for testing)
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

# Cosine similarity helper
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
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=doc
        )
        doc_embeddings.append(response.data[0].embedding)

    # Get embedding for query
    query_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_embedding = query_response.data[0].embedding

    # Compute cosine similarities
    similarities = [
        (doc, cosine_similarity(query_embedding, emb))
        for doc, emb in zip(docs, doc_embeddings)
    ]

    # Sort by similarity score (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return top 3 matches
    top_matches = [doc for doc, score in similarities[:3]]

    return {"matches": top_matches}
