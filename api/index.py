import os
import io
import time
import math
import heapq
import logging
from typing import List, Optional

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from openai import OpenAI
from pypdf import PdfReader

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====== Config via variables d'env ======
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_PROJECT = os.getenv("OPENAI_PROJECT", "")
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
  <title>Chat-PDF (Vercel)</title>
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
  <h1>📄 Chat-PDF — Vercel (FastAPI, 1 fichier)</h1>
  <p class="muted">Importe 1..N PDF et pose une question. Réponse basée sur les passages les plus pertinents.</p>

  <!-- Formulaire postant vers la même URL -->
  <form method="post" enctype="multipart/form-data" class="row">
    <label>
      <div>PDF(s) :</div>
      <input name="files" type="file" accept="application/pdf" multiple required />
    </label>

    <label>
      <div>Question :</div>
      <textarea name="question" placeholder="Ex. Résume les conclusions et liste 3 actions clés." required>{question_value}</textarea>
    </label>

    <button type="submit">Analyser & Répondre</button>
  </form>

  {status_block}
  {result_block}

  <p class="muted">Debug: <a href="/api/health">/api/health</a> • <a href="/api/routes">/api/routes</a></p>
</body>
</html>"""

def page(question_value: str = "", chunks_count: Optional[int] = None, answer: Optional[str] = None, error: Optional[str] = None):
    blocks = []
    if chunks_count is not None:
        blocks.append(f'<div class="ok">Indexation : {chunks_count} segments.</div>')
    if error:
        blocks.append(f'<div class="err">{error}</div>')
    status_block = "".join(blocks)
    result_block = f'<div class="card"><b>Réponse :</b>\n\n{answer}</div>' if answer else ""
    return HTML_PAGE.replace("{question_value}", question_value).replace("{status_block}", status_block).replace("{result_block}", result_block)

# ========== Helpers PDF/Chunks ==========
def chunk_text(text: str, chunk_size: int = 1400, overlap: int = 320) -> List[str]:
    paragraphs = (text or "").replace("\r", "\n").split("\n")
    chunks: List[str] = []
    cur = ""
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if len(cur) + len(p) + 1 <= chunk_size:
            cur = (cur + "\n" + p).strip()
        else:
            if cur:
                chunks.append(cur)
            tail = cur[-overlap:] if overlap > 0 and len(cur) > overlap else cur
            cur = (tail + "\n" + p).strip()
    if cur:
        chunks.append(cur)
    return [c for c in chunks if c.strip()]

# ========== Embeddings & Cosine ==========
def l2_norm(v: List[float]) -> float:
    s = sum(x * x for x in v)
    return math.sqrt(s) if s > 0 else 1e-12

def normalize(v: List[float]) -> List[float]:
    n = l2_norm(v)
    return [x / n for x in v]

def dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

def embed_texts(texts: List[str], batch_size: int = 80, pause_s: float = 0.0) -> List[List[float]]:
    vecs: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=batch)
        for item in resp.data:
            vecs.append(item.embedding)
        if pause_s > 0:
            time.sleep(pause_s)
    return vecs

def top_k_context(question: str, chunks: List[str], emb_matrix: List[List[float]], k: int = 4) -> List[str]:
    if not emb_matrix:
        return []
    q_resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=[question])
    q = q_resp.data[0].embedding
    qn = normalize(q)
    normed = [normalize(vec) for vec in emb_matrix]
    heap = []
    for idx, vec in enumerate(normed):
        score = dot(qn, vec)
        if len(heap) < k:
            heapq.heappush(heap, (score, idx))
        else:
            if score > heap[0][0]:
                heapq.heapreplace(heap, (score, idx))
    top = sorted(heap, key=lambda x: -x[0])
    return [chunks[i] for _, i in top]

# ========== LLM ==========
def answer_with_context(question: str, context_chunks: List[str]) -> str:
    joined = "\n\n---\n\n".join(context_chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Contexte:\n{joined}\n\nQuestion: {question}"},
    ]
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.2,
    )
    try:
        return resp.choices[0].message.content or ""
    except Exception as e:
        logger.exception("Erreur parsing LLM response: %s", e)
        return ""

# ========== Routes ==========
@app.get("/api/health")
async def health():
    return "OK"

@app.get("/api/routes")
async def routes():
    rows = []
    for r in app.router.routes:
        methods = sorted(list(r.methods)) if getattr(r, "methods", None) else []
        rows.append(f"{getattr(r, 'path', str(r))} — {', '.join(methods)}")
    return HTMLResponse("<br>".join(rows))

MAX_TOTAL_BYTES = 4_900_000

@app.get("/", response_class=HTMLResponse)
@app.post("/", response_class=HTMLResponse)
async def home(request: Request, files: List[UploadFile] = File(None), question: str = Form("")):
    # Vérification des clés API
    if not OPENAI_API_KEY:
        return page(error="OPENAI_API_KEY manquante.")
    if OPENAI_API_KEY.startswith("sk-proj-") and not OPENAI_PROJECT:
        return page(error="Clé `sk-proj-…` sans OPENAI_PROJECT=proj_xxx.")

    # Si méthode GET, afficher le formulaire vide
    if request.method == "GET":
        return page()

    try:
        # Vérification des fichiers
        if not files:
            return page(question_value=question, error="Veuillez sélectionner au moins un fichier PDF.")

        # Lecture des fichiers
        total_size = 0
        file_bytes_list: List[bytes] = []
        for f in files:
            b = await f.read()
            file_bytes_list.append(b)
            total_size += len(b)
            await f.close()

        if total_size > MAX_TOTAL_BYTES:
            return page(question_value=question, error="Fichiers trop volumineux pour Vercel (≈5MB max).")

        # Extraction du texte
        text = ""
        for b in file_bytes_list:
            try:
                reader = PdfReader(io.BytesIO(b))
                for page in reader.pages:
                    text += page.extract_text() or ""
            except Exception as e:
                logger.exception("Erreur parsing PDF: %s", e)

        chunks = chunk_text(text)
        if not chunks:
            return page(question_value=question, chunks_count=0, error="Aucun texte exploitable trouvé dans les PDF.")

        # Génération des embeddings et réponse
        emb_matrix = embed_texts(chunks, batch_size=60)
        ctx = top_k_context(question, chunks, emb_matrix, k=4)
        ans = answer_with_context(question, ctx)
        return page(question_value=question, chunks_count=len(chunks), answer=ans)

    except Exception as e:
        logger.exception("Erreur serveur: %s", e)
        return page(question_value=question, error=f"Erreur serveur : {e}")
