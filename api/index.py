import os, io, time
import numpy as np
from typing import List
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from openai import OpenAI
from PyPDF2 import PdfReader

# ========= Config OpenAI via variables d'env (Vercel Settings → Environment Variables) =========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_PROJECT = os.getenv("OPENAI_PROJECT", "")  # requis si ta clé est sk-proj-...
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

client_kwargs = {"api_key": OPENAI_API_KEY}
if OPENAI_PROJECT:
    client_kwargs["project"] = OPENAI_PROJECT
client = OpenAI(**client_kwargs)

app = FastAPI()

SYSTEM_PROMPT = (
    "Tu es un assistant qui répond uniquement à partir du contexte fourni (extraits de PDF). "
    "Si l'information n'est pas présente dans le contexte, dis-le clairement."
)

HTML_PAGE = """<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Chat-PDF (Vercel, 1 fichier)</title>
  <style>
    body{font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;margin:24px;max-width:960px}
    form{border:1px solid #e5e7eb;padding:16px;border-radius:12px}
    h1{margin-top:0}
    .row{display:grid;gap:16px}
    textarea{width:100%;min-height:120px;padding:8px}
    input[type=file]{padding:8px}
    button{background:#111827;color:#fff;border:0;padding:10px 14px;border-radius:8px;cursor:pointer}
    .card{border:1px solid #e5e7eb;padding:16px;border-radius:12px;margin-top:16px;white-space:pre-wrap}
    .ok{background:#ecfdf5;border:1px solid #10b981;padding:8px 12px;border-radius:8px;display:inline-block;margin:8px 0}
    .err{background:#fef2f2;border:1px solid #ef4444;padding:8px 12px;border-radius:8px;display:inline-block;margin:8px 0}
    .muted{color:#6b7280;font-size:12px}
  </style>
</head>
<body>
  <h1>📄 Chat-PDF — Vercel (FastAPI, fichier unique)</h1>
  <p class="muted">Importe 1..N PDF et pose une question. Réponse basée sur les passages les plus pertinents.</p>

  <form method="post" action="/ask" enctype="multipart/form-data" class="row">
    <label>
      <div>PDF(s) :</div>
      <input name="files" type="file" accept="application/pdf" multiple required />
    </label>

    <label>
      <div>Question :</div>
      <textarea name="question" placeholder="Ex. Résume les conclusions et liste 3 actions clés." required></textarea>
    </label>

    <button type="submit">Analyser & Répondre</button>
  </form>

  {status_block}

  {result_block}
</body>
</html>"""

def page(status: str = "", chunks_count: int | None = None, answer: str | None = None, error: str | None = None):
    blocks = []
    if chunks_count is not None:
        blocks.append(f'<div class="ok">Indexation : {chunks_count} segments.</div>')
    if error:
        blocks.append(f'<div class="err">{error}</div>')
    status_block = "".join(blocks)

    result_block = f'<div class="card"><b>Réponse :</b>\\n\\n{answer}</div>' if answer else ""
    return HTMLResponse(HTML_PAGE.format(status_block=status_block, result_block=result_block))

# ========= Utils =========
def read_pdfs(files: List[UploadFile]) -> str:
    txt = ""
    for uf in files:
        data = uf.file.read()
        uf.file.seek(0)
        reader = PdfReader(io.BytesIO(data))
        for page in reader.pages:
            txt += page.extract_text() or ""
    return txt

def chunk_text(text: str, chunk_size: int = 1400, overlap: int = 320) -> List[str]:
    paragraphs = (text or "").replace("\\r", "\\n").split("\\n")
    chunks, cur = [], ""
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if len(cur) + len(p) + 1 <= chunk_size:
            cur = (cur + "\\n" + p).strip()
        else:
            if cur:
                chunks.append(cur)
            tail = cur[-overlap:] if overlap > 0 else ""
            cur = (tail + "\\n" + p).strip()
    if cur:
        chunks.append(cur)
    return [c for c in chunks if c.strip()]

def embed_texts(texts: List[str], batch_size: int = 80, pause_s: float = 0.0) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=batch)
        for item in resp.data:
            vecs.append(item.embedding)
        if pause_s > 0:
            time.sleep(pause_s)
    return np.asarray(vecs, dtype=np.float32)

def top_k_context(question: str, chunks: List[str], emb_matrix: np.ndarray, k: int = 4) -> List[str]:
    q = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=[question]).data[0].embedding
    q = np.asarray(q, dtype=np.float32)
    emb_norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-12
    M = emb_matrix / emb_norms
    q_norm = q / (np.linalg.norm(q) + 1e-12)
    scores = M @ q_norm  # (n,)
    top_idx = np.argsort(-scores)[:k]
    return [chunks[i] for i in top_idx]

def answer_with_context(question: str, context_chunks: List[str]) -> str:
    joined = "\\n\\n---\\n\\n".join(context_chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Contexte:\\n{joined}\\n\\nQuestion: {question}"},
    ]
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content

# ========= Routes =========
@app.get("/", response_class=HTMLResponse)
async def home(_: Request):
    # Check clé
    if not OPENAI_API_KEY:
        return page(error="OPENAI_API_KEY manquante (ajoute-la dans Vercel → Settings → Environment Variables).")
    if OPENAI_API_KEY.startswith("sk-proj-") and not OPENAI_PROJECT:
        return page(error="Clé `sk-proj-…` détectée : ajoute aussi OPENAI_PROJECT=proj_xxx dans les variables d'environnement.")
    return page()

@app.post("/ask", response_class=HTMLResponse)
async def ask(
    request: Request,
    question: str = Form(...),
    files: List[UploadFile] = File(...)
):
    if not OPENAI_API_KEY:
        return page(error="OPENAI_API_KEY manquante.")
    if OPENAI_API_KEY.startswith("sk-proj-") and not OPENAI_PROJECT:
        return page(error="Clé `sk-proj-…` sans OPENAI_PROJECT=proj_xxx.")

    try:
        text = read_pdfs(files)
        chunks = chunk_text(text)
        if not chunks:
            return page(chunks_count=0, error="Aucun texte exploitable trouvé dans les PDF.")

        emb_matrix = embed_texts(chunks, batch_size=60)
        ctx = top_k_context(question, chunks, emb_matrix, k=4)
        ans = answer_with_context(question, ctx)
        return page(chunks_count=len(chunks), answer=ans)
    except Exception as e:
        return page(error=f"Erreur : {e}")
