import os
from typing import List, Dict
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from langchain_community.chat_models import ChatOllama

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("COLLECTION_NAME", "crypto_mica_v1")

EMB_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b")

TOP_K = int(os.getenv("TOP_K", 12))
TOP_N = int(os.getenv("TOP_N", 5))
NORMALIZE = True  # normalize embeddings (cosine)

# -----------------------------------------------------------------------------
# Init clients/models (lazy init acceptable for small setup)
# -----------------------------------------------------------------------------
qdrant = QdrantClient(url=QDRANT_URL)
embedder = SentenceTransformer(EMB_MODEL)

tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
reranker = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL)
reranker.eval()

if LLM_PROVIDER != "ollama":
    raise NotImplementedError("Questo blueprint supporta solo LLM via Ollama per ora.")
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def is_e5(model_id: str) -> bool:
    mid = (model_id or "").lower()
    return "e5" in mid or "intfloat/multilingual-e5" in mid

def embed_query(text: str) -> List[float]:
    if is_e5(EMB_MODEL):
        text = "query: " + text
    v = embedder.encode(text, normalize_embeddings=NORMALIZE, convert_to_numpy=False)
    return list(v)

def format_context(hits) -> str:
    """Build the context string for the LLM with inline citations."""
    blocks = []
    for h in hits:
        p = h.payload
        title = p.get("title", "")
        cid = p.get("chunk_id")
        text = p.get("text", "")
        blocks.append(f"[SOURCE: {title} | chunk {cid}]\n{text}")
    return "\n\n---\n\n".join(blocks)

def rerank_hits(query: str, hits, top_n: int) -> List:
    """Cross-encoder rerank; expects hits with payload['text'] available."""
    pairs = [(query, h.payload.get("text", "")) for h in hits]
    with torch.no_grad():
        enc = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        scores = reranker(**enc).logits.squeeze(-1)
        # sort by descending score
        sorted_idx = torch.argsort(scores, descending=True).tolist()
    return [hits[i] for i in sorted_idx[:top_n]]

# -----------------------------------------------------------------------------
# Core API
# -----------------------------------------------------------------------------
SYS_PROMPT = (
    "Sei un assistente per domande su whitepaper crypto e sul Regolamento MiCA (UE). "
    "Rispondi SOLO usando i contenuti presenti nei CHUNK forniti. "
    "Se non trovi informazioni sufficienti, rispondi: 'Non ho evidenze sufficienti nei documenti indicizzati.' "
    "Includi SEMPRE citazioni in forma [SOURCE: <titolo> | chunk <id>]. "
)

def retrieve(query: str, top_k: int) -> List:
    qv = embed_query(query)
    res = qdrant.search(collection_name=COLLECTION, query_vector=qv, limit=top_k)
    return res

def answer(question: str) -> Dict:
    hits = retrieve(question, TOP_K)
    if not hits:
        return {"answer": "Non ho evidenze sufficienti nei documenti indicizzati.", "sources": []}

    # rerank (cross-encoder)
    hits = rerank_hits(question, hits, TOP_N)

    # build context
    context = format_context(hits)

    # Compose messages for Ollama chat
    user_msg = (
        f"Domanda: {question}\n\n"
        f"CHUNK:\n{context}\n\n"
        "Rispondi in modo conciso e includi sempre citazioni [SOURCE: titolo | chunk]."
    )

    resp = llm.invoke(
        [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": user_msg},
        ]
    )

    sources = [
        {
            "title": h.payload.get("title"),
            "doc_id": h.payload.get("doc_id"),
            "chunk_id": h.payload.get("chunk_id"),
            "source_url": h.payload.get("source_url"),
        }
        for h in hits
    ]
    return {"answer": resp.content, "sources": sources}
