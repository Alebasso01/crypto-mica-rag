import sys, requests
query = sys.argv[1] if len(sys.argv) > 1 else "Qual è l'obiettivo del MiCA?"
resp = requests.post("http://localhost:8000/query", json={"query": query}, timeout=120)
print(resp.json())