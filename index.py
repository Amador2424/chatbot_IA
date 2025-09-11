import os
import io
import time
import numpy as np
from typing import List
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from openai import OpenAI

# ========= Config OpenAI (via variables d'environnement Vercel) =========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_PROJECT = os.getenv("OPENAI_PROJECT", "")  # requis si ta clé est sk-proj-...
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

client_kwargs = {"api_key": OPENAI_API_KEY}
if OPENAI_PROJECT:
    client_kwargs["project"] = OPENAI_PROJECT
client = OpenAI(**client_kwargs)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

SYSTEM_PROMPT = (
    "Tu es un assistant qui répond uniquement à partir du contexte fourni (extraits de PDF). "
    "Si l'information n'est pas présente dans le contexte, dis-le clairement."
)

# ========= Utils =========
def read_pdfs(files: List[UploadFile]) -> str:
    """Lit tout le texte de plusieurs PDFs."""
    from PyPDF2 import PdfReader
    txt = ""
    for uf in files:
        data = uf.file.read()
        uf.file.seek(0)
        reader = PdfReader(io.BytesIO(data))
        for page in reader.pages:
            txt += page.extract_text() or ""
    return txt

def chunk_text(text: str, chunk_size: int = 1400, overlap: int = 320) -> List[str]:
    """Découpe naïvement en chunks (avec recouvrement)."""
    paragraphs = (text or "").replace("\r", "\n").split("\n")
    chunks, cur = [], ""
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if len(cur) + len(p) + 1 <= chunk_size:
            cur = (cur + "\n" + p).strip()
        else:
            if cur:
                chunks.append(cur)
            tail = cur[-overlap:] if overlap > 0 else ""
            cur = (tail + "\n" + p).strip()
    if cur:
        chunks.append(cur)
    return [c for c in chunks if c.strip()]

def embed_texts(texts: List[str], batch_size: int = 80, pause_s: float = 0.0) -> np.ndarray:
    """Embeddings OpenAI (SDK officiel). Retourne (n,d)."""
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
    """Retourne les k chunks les plus pertinents (cosine)."""
    q = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=[question]).data[0].embedding
    q = np.asarray(q, dtype=np.float32)
    # cos sim = (A·B) / (||A|| ||B||). On normalise d'abord la matrice et le vecteur.
    emb_norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-12
    M = emb_matrix / emb_norms
    q_norm = q / (np.linalg.norm(q) + 1e-12)
    scores = M @ q_norm  # (n,)
    top_idx = np.argsort(-scores)[:k]
    return [chunks[i] for i in top_idx]

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
    return resp.choices[0].message.content

# ========= Routes =========
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "answer": None,
            "error": None,
            "chunks_count": None
        }
    )

@app.post("/ask", response_class=HTMLResponse)
async def ask(
    request: Request,
    question: str = Form(...),
    files: List[UploadFile] = File(...)
):
    # Sécurité clé
    if not OPENAI_API_KEY:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "answer": None,
            "error": "OPENAI_API_KEY manquante (ajoute-la en Variable d'environnement sur Vercel).",
            "chunks_count": None
        })

    try:
        text = read_pdfs(files)
        chunks = chunk_text(text)
        if not chunks:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "answer": None,
                "error": "Aucun texte exploitable dans les PDF.",
                "chunks_count": 0
            })

        emb_matrix = embed_texts(chunks, batch_size=60)
        ctx = top_k_context(question, chunks, emb_matrix, k=4)
        ans = answer_with_context(question, ctx)

        return templates.TemplateResponse("index.html", {
            "request": request,
            "answer": ans,
            "error": None,
            "chunks_count": len(chunks)
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "answer": None,
            "error": f"Erreur : {e}",
            "chunks_count": None
        })
