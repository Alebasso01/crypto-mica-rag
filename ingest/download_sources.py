import pathlib, requests, re

HEADERS = {"User-Agent": "Mozilla/5.0"}

def safe_filename(s: str) -> str:
    s = re.sub(r'[\\/:*?"<>|]+', '_', s)
    s = re.sub(r'\s+', '_', s).strip('_')
    return s.lower()

SEEDS = [
    {"title":"Bitcoin Whitepaper",
     "urls":[ "https://bitcoin.org/bitcoin.pdf" ]},
    {"title":"Ethereum Yellow Paper",
     "urls":[ "https://ethereum.github.io/yellowpaper/paper.pdf" ]},
    {"title":"Solana Whitepaper",
     "urls":[ "https://solana.com/solana-whitepaper.pdf" ]},
    {"title":"Polkadot Whitepaper",
     "urls":[
         "https://assets.polkadot.network/Polkadot-lightpaper.pdf",
         "https://www.allcryptowhitepapers.com/wp-content/uploads/2019/08/PolkaDotPaper.pdf",
         "https://raw.githubusercontent.com/polkadot-io/polkadot-white-paper/master/PolkaDotPaper.pdf"
     ]},
    {"title":"MiCA Regulation (EU) 2023/1114",
     "urls":[
         # PDF diretto (spesso ok)
         "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32023R1114&from=EN",
         # versione HTML (se il PDF non arriva)
         "https://eur-lex.europa.eu/eli/reg/2023/1114/oj"
     ]},
]

RAW = pathlib.Path("data/raw"); RAW.mkdir(parents=True, exist_ok=True)

def is_pdf(response: requests.Response, url: str, content: bytes) -> bool:
    ct = response.headers.get("Content-Type","").lower()
    return ("pdf" in ct) or url.lower().endswith(".pdf") or content[:4] == b"%PDF"

def download_first_ok(title: str, urls: list[str]) -> pathlib.Path:
    last_err = None
    for u in urls:
        try:
            r = requests.get(u, timeout=90, headers=HEADERS, allow_redirects=True)
            r.raise_for_status()
            content = r.content
            pdf = is_pdf(r, u, content)
            ext = ".pdf" if pdf else ".html"
            out = RAW / (safe_filename(title) + ext)
            out.write_bytes(content)
            print(f"Saved: {out} from {u} ({'PDF' if pdf else 'HTML'})")
            return out
        except Exception as e:
            last_err = e
            print("  fallback:", u, "->", type(e).__name__)
    if last_err:
        raise last_err

for s in SEEDS:
    download_first_ok(s["title"], s["urls"])
