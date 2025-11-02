import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
import requests


rows = []
with open("eval/golden_set.jsonl","r") as f:
    for line in f:
        item = json.loads(line)
        r = requests.post("http://localhost:8000/query", json={"query": item["question"]}, timeout=180)
        out = r.json()
        rows.append({
        "question": item["question"],
        "answer": out.get("answer",""),
        "contexts": [s.get("title","?") for s in out.get("sources",[])],
        "ground_truth": item["ground_truth"],
        })


report = evaluate(Dataset.from_list(rows), metrics=[faithfulness, answer_relevancy, context_precision])
print(report)