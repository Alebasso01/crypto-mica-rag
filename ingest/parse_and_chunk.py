import json, pathlib
from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter

RAW = pathlib.Path("data/raw")
OUT = pathlib.Path("data/processed"); OUT.mkdir(parents=True, exist_ok=True)

conv = DocumentConverter()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1800, chunk_overlap=300,
    separators=["\n\n", "\n", ". ", " "],
)

def title_injection(title: str, text: str) -> str:
    return f"[DOC_TITLE: {title}]\n{text}"

# << cambia questa parte: prendiamo sia .pdf che .html >>
files = [p for p in RAW.iterdir() if p.suffix.lower() in {".pdf", ".html"}]

for path in files:
    title = path.stem.replace("_"," ").title()
    doc = conv.convert(path.as_posix())   # Docling supporta PDF/HTML
    text = doc.document.export_to_text()
    chunks = splitter.split_text(text)
    with open(OUT / (path.stem + ".jsonl"), "w", encoding="utf-8") as f:
        for i, c in enumerate(chunks):
            rec = {
                "doc_id": path.name,
                "title": title,
                "chunk_id": i,
                "text": title_injection(title, c),
                "source_url": None,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("Chunks:", path.name, len(chunks))
