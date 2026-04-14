from llama_index.core.llms import ChatMessage
from llama_index.llms.groq import Groq

from app.core.config import settings
from app.rag.retriever import build_retriever

# ── initialise LLM once at module level ──────────────────────────
_retriever = None

_llm = Groq(
    model="llama-3.3-70b-versatile",
    api_key=settings.GROQ_API_KEY,
)

_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY "
    "the context provided below. If the context doesn't contain enough "
    "information, say so clearly.\n\n"
    "Context:\n{context}"
)


def _get_retriever():
    """Lazy-init: build the retriever on first query (not at import time)."""
    global _retriever
    if _retriever is None:
        _retriever = build_retriever()
    return _retriever


def query_pipeline(query: str) -> dict:
    """Retrieve relevant nodes and generate an LLM answer.

    Returns
    -------
    dict with keys: answer (str), contexts (list[str])
    """

    # 1. Retrieve (hybrid + rerank)
    nodes = _get_retriever().retrieve(query)
    contexts = [n.get_content() for n in nodes]
    context = "\n\n---\n\n".join(contexts)

    # 2. Build messages
    messages = [
        ChatMessage(role="system", content=_SYSTEM_PROMPT.format(context=context)),
        ChatMessage(role="user", content=query),
    ]

    # 3. Generate
    response = _llm.chat(messages)
    return {"answer": response.message.content, "contexts": contexts}
