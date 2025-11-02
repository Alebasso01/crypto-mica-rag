import os
import json
import uuid
import pathlib
from typing import List
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("COLLECTION_NAME", "crypto_mica_v1")
EMB_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

DATA_DIR = pathlib.Path("data/processed")

BATCH_SIZE = 128  # embedding & upsert batch
NORMALIZE = True  # normalize embeddings for cosine

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def is_e5(model_id: str) -> bool:
    mid = (model_id or "").lower()
    return "e5" in mid or "intfloat/multilingual-e5" in mid

def embed_passages(model: SentenceTransformer, texts: List[str]) -> List[List[float]]:
    # E5 vuole il prefisso "passage: "
    if is_e5(EMB_MODEL):
        texts = [("passage: " + t) for t in texts]
    return model.encode(texts, normalize_embeddings=NORMALIZE, convert_to_numpy=False)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # 1) init client & model
    client = QdrantClient(url=QDRANT_URL)
    embedder = SentenceTransformer(EMB_MODEL)
    dim = embedder.get_sentence_embedding_dimension()

    # 2) crea collection se non esiste
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        print(f"[Qdrant] Created collection '{COLLECTION}' (dim={dim})")
    else:
        print(f"[Qdrant] Using existing collection '{COLLECTION}' (dim={dim})")

    # 3) carica jsonl e upsert a batch
    files = sorted(DATA_DIR.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"Nessun file .jsonl in {DATA_DIR}. Esegui prima parse_and_chunk.py")

    total_points = 0
    for fp in files:
        records = []
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                # rec: {doc_id,title,chunk_id,text,source_url}
                records.append(rec)

        # batched
        for start in range(0, len(records), BATCH_SIZE):
            batch = records[start : start + BATCH_SIZE]
            texts = [r["text"] for r in batch]
            vecs = embed_passages(embedder, texts)

            points = []
            for r, v in zip(batch, vecs):
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=list(v),
                        payload={
                            "doc_id": r.get("doc_id"),
                            "title": r.get("title"),
                            "chunk_id": r.get("chunk_id"),
                            "text": r.get("text"),
                            "source_url": r.get("source_url"),
                        },
                    )
                )

            client.upsert(collection_name=COLLECTION, points=points)
            total_points += len(points)
            print(f"[Upsert] {fp.name} +{len(points)} (tot={total_points})")

    print(f"✅ Completato. Punti totali indicizzati: {total_points}")

if __name__ == "__main__":
    main()
