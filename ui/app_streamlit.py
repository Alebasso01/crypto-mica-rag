import streamlit as st
import requests


st.set_page_config(page_title="Crypto+MiCA RAG", layout="centered")
st.title("🔎 Crypto + MiCA RAG (locale)")


q = st.text_input("Fai una domanda…", "Qual è l'obiettivo del MiCA?")
if st.button("Chiedi"):
    r = requests.post("http://localhost:8000/query", json={"query": q}, timeout=120)
    data = r.json()
    st.markdown("### Risposta")
    st.write(data.get("answer"))
    st.markdown("### Fonti")
    for s in data.get("sources", []):
        st.write(f"• {s['title']} (chunk {s['chunk_id']})")