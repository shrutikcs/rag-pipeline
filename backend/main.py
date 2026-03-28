from fastapi import FastAPI

app = FastAPI(title="FastAPI RAG Backend")


@app.get("/")
async def root():
    return {"message": "Hello World - RAG Backend is live!"}
