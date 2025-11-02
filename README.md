# Crypto + MiCA RAG (100% free & local)


RAG locale su whitepaper crypto + Regolamento MiCA (UE). Nessun servizio a pagamento: Qdrant OSS, bge-m3, bge-reranker, Ollama (Mistral/Llama).


## Stack
- Parsing: Docling (OSS)
- Embeddings: BAAI/bge-m3 (locale)
- Vector DB: Qdrant (Docker)
- Reranker: BAAI/bge-reranker-base (locale, CPU ok)
- LLM: Ollama `mistral:7b` (o `llama3.1:8b-instruct`)
- Serving: FastAPI


## Setup rapido
```bash
# 1) Python venv
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt


# 2) Qdrant (Docker)
docker compose up -d


# 3) Ollama + modello
curl -fsSL https://ollama.com/install.sh | sh # (Linux/macOS)
ollama pull mistral:7b


# 4) Dati + Index
touch .env && cp .env.example .env
python ingest/download_sources.py
python ingest/parse_and_chunk.py
python ingest/upsert_qdrant.py


# 5) API
uvicorn service.api:app --reload --port 8000


# 6) Test
python ui/cli_test.py "Qual è l'obiettivo del MiCA?"