from pathlib import Path

from sqlalchemy import make_url

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

from app.core.config import settings

# ── paths ──────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# ── embedding model ───────────────────────────────────────────────
embed_model = GoogleGenAIEmbedding(
    api_key=settings.GEMINI_API_KEY,
    model_name="gemini-embedding-001",
)


# ── vector store (Neon pgvector) ──────────────────────────────────
def _make_vector_store() -> PGVectorStore:
    url = make_url(settings.DATABASE_URL)
    return PGVectorStore.from_params(
        database=url.database,
        host=url.host,
        password=url.password,
        port=str(url.port or 5432),
        user=url.username,
        table_name="document_vectors",
        embed_dim=3072,  # gemini-embedding-001 output dimension
    )


# ── ingest pipeline ──────────────────────────────────────────────
def ingest() -> VectorStoreIndex:
    """Load docs from data/, chunk, embed, and store in pgvector."""

    # 1. Load .txt and .pdf files
    if not any(DATA_DIR.glob("*.[tp][xd][tf]")):
        raise FileNotFoundError(
            f"No .txt or .pdf files in {DATA_DIR}. Add documents before ingesting."
        )

    reader = SimpleDirectoryReader(
        input_dir=str(DATA_DIR),
        required_exts=[".txt", ".pdf"],
    )
    documents = reader.load_data()
    print(f"Loaded {len(documents)} document(s) from {DATA_DIR}")

    # 2. Chunk with SentenceSplitter
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"Split into {len(nodes)} chunk(s)")

    # 3. Store in pgvector (embedding happens automatically)
    vector_store = _make_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(  # iterates through every node, calls gemini api for each chunch to turn into vector, uploads the vector and original text to postgres, creates the searchable index.
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    print("Ingestion complete — vectors stored in Neon pgvector.")
    return index


if __name__ == "__main__":
    ingest()
