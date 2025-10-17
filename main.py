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
    """
    Accepts a query and a list of documents, returns the top 3 most similar documents.
    """
    docs = req.docs
    query = req.query
    
    if not docs or not query:
        raise HTTPException(status_code=400, detail="Both 'docs' and 'query' must be provided.")
    
    try:
        # --- SEPARATE EMBEDDINGS FOR QUERY AND DOCUMENTS ---
        # Embed the query with RETRIEVAL_QUERY task type
        query_result = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="RETRIEVAL_QUERY"  # Use RETRIEVAL_QUERY for the query
        )
        query_emb = query_result["embedding"]
        
        # Embed the documents with RETRIEVAL_DOCUMENT task type
        docs_result = genai.embed_content(
            model="models/text-embedding-004",
            content=docs,
            task_type="RETRIEVAL_DOCUMENT"  # Use RETRIEVAL_DOCUMENT for documents
        )
        doc_embeddings = docs_result["embedding"]
        
        # --- SIMILARITY COMPUTATION ---
        similarities = [
            (doc, cosine_similarity(query_emb, emb))
            for doc, emb in zip(docs, doc_embeddings)
        ]
        
        # Sort by similarity score in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Extract the text of the top 3 matches
        top_matches = [doc for doc, _ in similarities[:3]]
        
        return {"matches": top_matches}
    
    # --- ERROR HANDLING ---
    except exceptions.GoogleAPICallError as e:
        raise HTTPException(status_code=500, detail=f"Google API Error: {e.message}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
