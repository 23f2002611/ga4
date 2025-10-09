from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import google.generativeai as genai
import os
from google.api_core import exceptions

# --- Configuration ---
# Make sure you have your GEMINI_API_KEY set as an environment variable
# For example: export GEMINI_API_KEY="YOUR_API_KEY"
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
        # --- BATCH EMBEDDING ---
        # Combine query and docs for a single, efficient API call
        content_to_embed = [query] + docs
        
        # Call the API once for all content
        embedding_result = genai.embed_content(
            model="models/text-embedding-004",  # Use the latest recommended model
            content=content_to_embed,
            task_type="RETRIEVAL_DOCUMENT" # Specify the task type for better performance
        )

        embeddings = embedding_result["embedding"]
        
        # Separate the query embedding from the document embeddings
        query_emb = embeddings[0]
        doc_embeddings = embeddings[1:]

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
        # Catch specific API errors (like permission denied, invalid argument)
        raise HTTPException(status_code=500, detail=f"Google API Error: {e.message}")
    except Exception as e:
        # Catch any other unexpected errors
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
        
