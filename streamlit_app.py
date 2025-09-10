import os
import io
import math
import time
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from typing import List, Tuple
from PyPDF2 import PdfReader
from openai import OpenAI

# =========================
#   CONFIG & SECRETS
# =========================
st.set_page_config(page_title="Chat-PDF (OpenAI SDK + FAISS)", page_icon="📄", layout="wide")
load_dotenv()  # charge .env en local si présent

def _get_secret(name: str, default: str | None = None) -> str | None:
    val = st.secrets.get(name) if hasattr(st, "secrets") else None
    if not val:
        val = os.getenv(name)
    if isinstance(val, str):
        val = val.strip().strip('"').strip("'")
    return val or default

OPENAI_API_KEY = _get_secret("OPENAI_API_KEY")
OPENAI_PROJECT = _get_secret("OPENAI_PROJECT")  # requis si ta clé est sk-proj-…
OPENAI_MODEL = _get_secret("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = _get_secret("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY manquante. Ajoute-la dans `.env` ou dans Settings → Secrets.")
    st.stop()

# Client OpenAI (le SDK officiel supporte les sk-proj-…)
client_kwargs = {"api_key": OPENAI_API_KEY}
if OPENAI_PROJECT:
    client_kwargs["project"] = OPENAI_PROJECT
client = OpenAI(**client_kwargs)

with st.sidebar:
    st.caption("🔧 Configuration OpenAI (SDK officiel)")
    st.write("Clé :", "✅ détectée" if OPENAI_API_KEY else "❌ absente")
    st.write("Type :", "sk-proj-…" if OPENAI_API_KEY.startswith("sk-proj-") else "sk-…")
    st.write("Project ID :", OPENAI_PROJECT or ("—" if not OPENAI_API_KEY.startswith("sk-proj-") else "❌ requis"))
    st.write("LLM :", OPENAI_MODEL)
    st.write("Embeddings :", OPENAI_EMBED_MODEL)

# =========================
#   UTILS (PDF & CHUNKS)
# =========================
def extract_text_from_pdfs(files: List[io.BytesIO]) -> str:
    text = ""
    for f in files:
        reader = PdfReader(f)
        for page in reader.pages:
            text += (page.extract_text() or "")
    return text

def chunk_text(text: str, chunk_size: int = 1400, overlap: int = 320) -> List[str]:
    text = text.replace("\r", "\n")
    paragraphs = text.split("\n")
    chunks, current = [], ""
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if len(current) + len(p) + 1 <= chunk_size:
            current = (current + "\n" + p).strip()
        else:
            if current:
                chunks.append(current)
            # overlap simple: on garde la fin de current
            tail = current[-overlap:] if overlap > 0 else ""
            current = (tail + "\n" + p).strip()
    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]

# =========================
#   EMBEDDINGS & FAISS
# =========================
def embed_texts(texts: List[str], batch_size: int = 90, sleep_s: float = 0.0) -> np.ndarray:
    """
    Embed via OpenAI SDK officiel (supporte sk-proj-) — renvoie un np.ndarray (n, d)
    """
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=batch)
        # l'API renvoie dans le même ordre
        for item in resp.data:
            vecs.append(item.embedding)
        if sleep_s > 0:
            time.sleep(sleep_s)
    return np.array(vecs, dtype=np.float32)

def build_faiss_index(embeddings: np.ndarray):
    """
    Construit un index FAISS cosine (via normalisation + Inner Product).
    """
    import faiss  # import ici pour éviter erreurs si package manquant
    # normalisation L2 -> cos sim = dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    normed = embeddings / norms
    d = normed.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product
    index.add(normed)
    return index

def search_index(index, embeddings_matrix: np.ndarray, query: str, texts: List[str], k: int = 4) -> List[Tuple[int, float, str]]:
    """
    Embedding de la question -> recherche FAISS -> renvoie [(idx, score, chunk_text), ...]
    """
    import faiss
    q_vec = embed_texts([query])
    q_norm = q_vec / (np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-12)
    scores, indices = index.search(q_norm, k)
    results = []
    for rank in range(indices.shape[1]):
        idx = int(indices[0, rank])
        if idx == -1:
            continue
        score = float(scores[0, rank])
        results.append((idx, score, texts[idx]))
    return results

# =========================
#   CHAT COMPLETIONS
# =========================
SYSTEM_PROMPT = (
    "Tu es un assistant qui répond strictement à partir du contexte fourni, issu de PDF de l'utilisateur. "
    "Si l'information n'est pas présente dans le contexte, dis-le clairement. Cite ou paraphrase précisément."
)

def answer_with_context(question: str, context_chunks: List[str]) -> str:
    # On tronque proprement si nécessaire
    MAX_CONTEXT_CHARS = 9000
    joined = "\n\n---\n\n".join(context_chunks)
    if len(joined) > MAX_CONTEXT_CHARS:
        joined = joined[:MAX_CONTEXT_CHARS] + "…"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Contexte:\n{joined}\n\nQuestion: {question}"}
    ]
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content

# =========================
#   UI & STATE
# =========================
st.title("📄 Chat avec vos PDF — OpenAI SDK + FAISS (sans LangChain)")
st.caption("Téléverse des PDF, indexe-les (embeddings OpenAI), puis pose tes questions.")

if "pdf_ready" not in st.session_state:
    st.session_state.pdf_ready = False
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "emb_matrix" not in st.session_state:
    st.session_state.emb_matrix = None
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "history" not in st.session_state:
    st.session_state.history = []  # [(user, assistant)]

# Question input
user_q = st.text_input("Ta question sur le(s) PDF :", placeholder="Ex: Résume la section 2 et donne les actions clés…")

# Sidebar: upload + extraction
with st.sidebar:
    st.subheader("Étapes")
    st.markdown("1) Importer vos **PDF**  \n2) Cliquer **Extraction**  \n3) Poser vos questions")
    pdf_files = st.file_uploader("Importer des PDF", type=["pdf"], accept_multiple_files=True)

    if st.button("Extraction"):
        if not pdf_files:
            st.warning("Merci d'importer au moins un PDF.")
        else:
            with st.spinner("📚 Lecture et indexation…"):
                try:
                    text = extract_text_from_pdfs(pdf_files)
                    chunks = chunk_text(text, chunk_size=1400, overlap=320)
                    if not chunks:
                        st.error("Aucun texte exploitable trouvé dans les PDF.")
                    else:
                        vecs = embed_texts(chunks, batch_size=80)
                        idx = build_faiss_index(vecs)
                        st.session_state.chunks = chunks
                        st.session_state.emb_matrix = vecs
                        st.session_state.faiss_index = idx
                        st.session_state.pdf_ready = True
                        st.success(f"Indexation terminée ✅ ({len(chunks)} chunks)")
                except Exception as e:
                    st.error(f"Échec de l'indexation : {e}")

# Pose la question
if user_q:
    if not st.session_state.pdf_ready or st.session_state.faiss_index is None:
        st.info("📥 Importez vos PDF puis cliquez sur **Extraction** avant de poser une question.")
    else:
        try:
            results = search_index(
                st.session_state.faiss_index,
                st.session_state.emb_matrix,
                user_q,
                st.session_state.chunks,
                k=4
            )
            top_context = [c for _, _, c in results]
            answer = answer_with_context(user_q, top_context)
            st.session_state.history.append((user_q, answer))
        except Exception as e:
            st.error(f"Erreur pendant la génération de la réponse : {e}")

# Rendu simple du chat
if st.session_state.history:
    st.markdown("---")
    st.subheader("Historique")
    for i, (u, a) in enumerate(st.session_state.history):
        st.markdown(f"**👤 Toi :** {u}")
        st.markdown(f"**🤖 Assistant :** {a}")
        if i < len(st.session_state.history) - 1:
            st.markdown("---")
