import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from streamlit_chat import message

# =========================
#   CONFIG & SECRETS
# =========================
st.set_page_config(page_title="Chat-PDF", page_icon="📄", layout="wide")
load_dotenv()  # charge .env si présent (local)

def _get_secret(name: str, default: str | None = None) -> str | None:
    val = st.secrets.get(name) if hasattr(st, "secrets") else None
    if not val:
        val = os.getenv(name)
    if isinstance(val, str):
        val = val.strip().strip('"').strip("'")
    return val or default

OPENAI_API_KEY = _get_secret("OPENAI_API_KEY")
OPENAI_PROJECT = _get_secret("OPENAI_PROJECT")  # requis si ta clé commence par sk-proj-
OPENAI_MODEL = _get_secret("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = _get_secret("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Export pour les SDKs sous-jacents
if OPENAI_API_KEY:  os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if OPENAI_PROJECT:  os.environ["OPENAI_PROJECT"] = OPENAI_PROJECT

# Panneau statut config
with st.sidebar:
    st.caption("🔧 Configuration OpenAI")
    st.write("Clé :", "✅ détectée" if OPENAI_API_KEY else "❌ absente")
    st.write("Type de clé :", "sk-proj-…" if (OPENAI_API_KEY or "").startswith("sk-proj-") else "sk-… / inconnu")
    st.write("Project ID :", OPENAI_PROJECT or ("—" if not (OPENAI_API_KEY or "").startswith("sk-proj-") else "❌ requis"))
    st.write("LLM :", OPENAI_MODEL)
    st.write("Embeddings :", OPENAI_EMBEDDING_MODEL)

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY manquante. Ajoute-la dans `.env` (local) ou Settings → Secrets (Cloud).")
    st.stop()

# Si clé sk-proj- sans Project ID -> 401 assuré
if OPENAI_API_KEY.startswith("sk-proj-") and not OPENAI_PROJECT:
    st.error("Ta clé commence par `sk-proj-` : ajoute aussi `OPENAI_PROJECT=proj_xxx` dans `.env` ou Secrets.")
    st.stop()

# =========================
#   HELPERS
# =========================
def extract_chunks_from_pdfs(pdf_docs):
    """Lit les PDFs, agrège le texte et crée des chunks."""
    full_text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            full_text += (page.extract_text() or "")

    if not full_text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1400,
        chunk_overlap=320,
        length_function=len,
    )
    return splitter.split_text(full_text)

def create_vectorstore(chunks):
    """Crée un index FAISS en mémoire à partir des chunks."""
    try:
        embeddings = OpenAIEmbeddings(
            api_key=OPENAI_API_KEY,
            model=OPENAI_EMBEDDING_MODEL,
            project=OPENAI_PROJECT  # utile pour sk-proj-…
        )
        return FAISS.from_texts(texts=chunks, embedding=embeddings)
    except Exception as e:
        raise RuntimeError(f"Échec création embeddings (clé/projet/modèle) : {e}")

def build_conversation_chain(vectorstore):
    """Chaîne conversationnelle avec mémoire + retrieval FAISS."""
    try:
        llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=OPENAI_MODEL,
            temperature=0.2,
            project=OPENAI_PROJECT  # utile pour sk-proj-…
        )
    except Exception as e:
        raise RuntimeError(f"Échec initialisation LLM (clé/projet/modèle) : {e}")

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
    )

def handle_user_question(user_text):
    if not st.session_state.get("pdf_ready", False):
        st.info("📥 Importez des PDF puis cliquez sur **Extraction**.")
        return
    if not user_text.strip():
        return
    try:
        result = st.session_state.conversation({"question": user_text})
        st.session_state.chat_history = result["chat_history"]
        for i, msg in enumerate(st.session_state.chat_history):
            if msg.type == "human":
                message(msg.content, is_user=True, key=f"u_{i}")
            else:
                message(msg.content, key=f"a_{i}")
    except Exception as e:
        st.error(f"Erreur pendant la génération de réponse : {e}")

# =========================
#   UI
# =========================
st.title("📄 Chat avec vos PDF")
st.caption("Téléversez un ou plusieurs PDF, indexez-les, puis posez vos questions en langage naturel.")

# States
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_ready" not in st.session_state:
    st.session_state.pdf_ready = False

# Input question
user_input = st.text_input(
    "Votre question sur le(s) PDF :",
    placeholder="Ex : Résume la section 2 et extrais les actions clés…"
)
if user_input:
    handle_user_question(user_input)

# Sidebar : upload & extraction
with st.sidebar:
    st.subheader("Étapes")
    st.markdown("1) Importer vos PDF\n\n2) Cliquer **Extraction**\n\n3) Poser vos questions")
    pdf_docs = st.file_uploader("Importer des PDF", type=["pdf"], accept_multiple_files=True)

    if st.button("Extraction"):
        if not pdf_docs:
            st.warning("Veuillez importer au moins un PDF.")
        else:
            with st.spinner("Indexation en cours…"):
                try:
                    chunks = extract_chunks_from_pdfs(pdf_docs)
                    if not chunks:
                        st.error("Aucun texte exploitable trouvé dans les PDF.")
                    else:
                        vs = create_vectorstore(chunks)
                        st.session_state.conversation = build_conversation_chain(vs)
                        st.session_state.pdf_ready = True
                        st.success(f"Indexation terminée ✅ ({len(chunks)} chunks)")
                except Exception as e:
                    st.error(f"Échec de l'indexation : {e}")

# Historique
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("Historique")
    for i, msg in enumerate(st.session_state.chat_history):
        if msg.type == "human":
            message(msg.content, is_user=True, key=f"hist_u_{i}")
        else:
            message(msg.content, key=f"hist_a_{i}")
