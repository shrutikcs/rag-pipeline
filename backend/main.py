from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from retrieval_raw import retrieve_raw

app = FastAPI(title="FastAPI RAG Backend")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchQuery(BaseModel):
    query: str

@app.get("/")
async def root():
    return {"message": "Hello World - RAG Backend is live!"}

@app.post("/search")
async def search(query_data: SearchQuery):
    results = await retrieve_raw(query_data.query)
    return {"results": results}
