import os
import io
import time
import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from openai import OpenAI

# ========================= 
#   CONFIG
# =========================
st.set_page_config(page_title="Chat-PDF", page_icon="📄", layout="wide")

# ⚠️ Clé + Project ID en dur (pas recommandé)
OPENAI_API_KEY = "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"
OPENAI_PROJECT = "proj_MpAHQtSkHp1U1SHeD0nR8TUO"   # <-- mets ton vrai Project ID ici
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_EMBED_MODEL = "text-embedding-3-small"

# Client OpenAI officiel (supporte les sk-proj-…)
client = OpenAI(api_key=OPENAI_API_KEY, project=OPENAI_PROJECT)

# =========================
#   UTILS (PDF + CHUNKS)
# =========================
def extract_text_from_pdfs(files):
    text = ""
    for f in files:
        reader = PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=1400, overlap=320):
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
            tail = current[-overlap:] if overlap > 0 else ""
            current = (tail + "\n" + p).strip()
    if current:
        chunks.append(current)
    return chunks

# =========================
#   EMBEDDINGS + FAISS
# =========================
def embed_texts(texts, batch_size=90):
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=batch)
        for item in resp.data:
            vecs.append(item.embedding)
        time.sleep(0.5)  # pour éviter le rate limit
    return np.array(vecs, dtype=np.float32)

def build_faiss_index(embeddings):
    import faiss
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    normed = embeddings / norms
    d = normed.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(normed)
    return index

def search_index(index, embeddings_matrix, query, texts, k=4):
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
#   CHAT COMPLETION
# =========================
SYSTEM_PROMPT = (
    "Tu es un assistant qui répond uniquement à partir du contexte fourni (extraits de PDF). "
    "Si l'information n'est pas présente dans le contexte, dis-le clairement."
)

def answer_with_context(question, context_chunks):
    joined = "\n\n---\n\n".join(context_chunks)
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
#   UI STREAMLIT
# =========================
st.title("📄 Chat avec vos PDF oui (OpenAI SDK + FAISS)")
st.caption("Upload des PDF → extraction du texte → embeddings → questions/réponses.")

if "pdf_ready" not in st.session_state:
    st.session_state.pdf_ready = False
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "emb_matrix" not in st.session_state:
    st.session_state.emb_matrix = None
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "history" not in st.session_state:
    st.session_state.history = []

# Input user
user_q = st.text_input("❓ Pose une question sur le(s) PDF :", placeholder="Ex: Résume la section 2...")

# Sidebar upload
with st.sidebar:
    st.subheader("Étapes")
    pdf_files = st.file_uploader("Importer des PDF", type=["pdf"], accept_multiple_files=True)

    if st.button("Extraction"):
        if not pdf_files:
            st.warning("⚠️ Importez au moins un PDF.")
        else:
            with st.spinner("📚 Lecture + indexation..."):
                try:
                    text = extract_text_from_pdfs(pdf_files)
                    chunks = chunk_text(text, chunk_size=1400, overlap=320)
                    vecs = embed_texts(chunks, batch_size=80)
                    idx = build_faiss_index(vecs)
                    st.session_state.chunks = chunks
                    st.session_state.emb_matrix = vecs
                    st.session_state.faiss_index = idx
                    st.session_state.pdf_ready = True
                    st.success(f"✅ Indexation terminée ({len(chunks)} chunks)")
                except Exception as e:
                    st.error(f"Erreur indexation : {e}")

# Si question posée
if user_q:
    if not st.session_state.pdf_ready:
        st.info("📥 Importez vos PDF puis cliquez sur Extraction avant de poser une question.")
    else:
        try:
            results = search_index(
                st.session_state.faiss_index,
                st.session_state.emb_matrix,
                user_q,
                st.session_state.chunks,
                k=4
            )
            context = [c for _, _, c in results]
            answer = answer_with_context(user_q, context)
            st.session_state.history.append((user_q, answer))
        except Exception as e:
            st.error(f"Erreur génération : {e}")

# Historique
if st.session_state.history:
    st.markdown("---")
    st.subheader("Historique")
    for i, (u, a) in enumerate(st.session_state.history):
        st.markdown(f"**👤 Toi :** {u}")
        st.markdown(f"**🤖 Assistant :** {a}")
        st.markdown("---")
