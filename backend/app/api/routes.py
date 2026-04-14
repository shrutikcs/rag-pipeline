from fastapi import APIRouter
from pydantic import BaseModel

from app.rag.ingest import ingest
from app.rag.pipeline import query_pipeline

router = APIRouter()


class QueryRequest(BaseModel):
    query: str


@router.post("/ingest")
def run_ingest():
    """Load documents from data/, chunk, embed, and store in pgvector."""
    ingest()
    return {"message": "Ingestion complete."}


@router.post("/query")
def run_query(body: QueryRequest):
    """Retrieve context and generate an LLM answer."""
    result = query_pipeline(body.query)
    return result  # {"answer": "...", "contexts": ["...", ...]}
