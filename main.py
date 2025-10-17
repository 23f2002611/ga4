from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import google.generativeai as genai
import os
from google.api_core import exceptions

# --- Configuration ---
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
except AttributeError:
    print("FATAL: GEMINI_API_KEY environment variable not set.")
    exit(1)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Semantic Similarity API",
    description="An API to find the most similar documents to a query using Gemini embeddings.",
    version="1.0.0",
)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

# --- Pydantic Schemas ---
class SimilarityRequest(BaseModel):
    docs: List[str]
    query: str

class SimilarityResponse(BaseModel):
    matches: List[str]

# --- Helper Function ---
def cosine_similarity(vec_a, vec_b):
    """Computes cosine similarity between two vectors."""
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

# --- API Endpoint ---
@app.post("/similarity", response_model=SimilarityResponse)
async def similarity_endpoint(req: SimilarityRequest):
    docs = req.docs
    query = req.query

    if not docs or not query:
        raise HTTPException(status_code=400, detail="Both 'docs' and 'query' must be provided.")

    try:
        # Embed the query
        query_result = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="RETRIEVAL_QUERY"
        )
        query_emb = np.array(query_result["embedding"])

        # Embed each document separately
        doc_embeddings = []
        for doc in docs:
            doc_result = genai.embed_content(
                model="models/text-embedding-004",
                content=doc,
                task_type="RETRIEVAL_DOCUMENT"
            )
            doc_embeddings.append(np.array(doc_result["embedding"]))

        # Compute cosine similarities
        similarities = [
            (doc, cosine_similarity(query_emb, emb))
            for doc, emb in zip(docs, doc_embeddings)
        ]

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_matches = [doc for doc, _ in similarities[:3]]

        return {"matches": top_matches}

    except exceptions.GoogleAPICallError as e:
        raise HTTPException(status_code=500, detail=f"Google API Error: {e.message}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

