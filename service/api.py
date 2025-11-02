from fastapi import FastAPI
from pydantic import BaseModel
from .rag_chain import answer


app = FastAPI(title="Crypto+MiCA RAG (local)")


class Query(BaseModel):
    query: str


@app.post("/query")
def query(q: Query):
    return answer(q.query)