import os
import streamlit as st
from dotenv import load_dotenv  # ✅ manquait
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from streamlit_chat import message

st.set_page_config(page_title="Chat-PDF", page_icon="📄", layout="wide")

# 1) Charger .env en local
load_dotenv()

# 2) Secrets (Cloud) puis .env (local)
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = (
    st.secrets.get("OPENAI_MODEL")
    or os.getenv("OPENAI_MODEL")
    or "gpt-4o-mini"
)
OPENAI_EMBEDDING_MODEL = (
    st.secrets.get("OPENAI_EMBEDDING_MODEL")
    or os.getenv("OPENAI_EMBEDDING_MODEL")
    or "text-embedding-3-small"
)

# 3) Sécuriser
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY manquante. Ajoute-la dans .env (local) ou Settings → Secrets (Cloud).")
    st.stop()

# --- Helpers
def extract_chunks_from_pdfs(pdf_docs):
    """Lit les PDFs, agrège le texte et crée des chunks."""
    full_text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            full_text += page_text

    if not full_text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1400,
        chunk_overlap=320,
        length_function=len,
    )
    chunks = splitter.split_text(full_text)
    return chunks

def create_vectorstore(chunks):
    """Crée un index FAISS en mémoire à partir des chunks."""
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model=OPENAI_EMBEDDING_MODEL)
    return FAISS.from_texts(texts=chunks, embedding=embeddings)

def build_conversation_chain(vectorstore):
    """Chaîne conversationnelle avec mémoire + retrieval FAISS."""
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=OPENAI_MODEL, temperature=0.2)  # ✅ OPENAI_MODEL
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory
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
        # Rendu chat
        for i, msg in enumerate(st.session_state.chat_history):
            if msg.type == "human":
                message(msg.content, is_user=True, key=f"u_{i}")
            else:
                message(msg.content, key=f"a_{i}")
    except Exception as e:
        st.error(f"Erreur pendant la génération de réponse : {e}")

# --- UI
st.title("📄 Chat avec vos PDF")
st.caption("Téléversez un ou plusieurs PDF, indexez-les, puis posez vos questions en langage naturel.")

# Init states
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_ready" not in st.session_state:
    st.session_state.pdf_ready = False

# Zone de question
user_input = st.text_input("Votre question sur le(s) PDF :", placeholder="Ex: Résume la section 2 et extrais les actions clés…")
if user_input:
    handle_user_question(user_input)

# Sidebar : import + extraction
with st.sidebar:
    st.subheader("Étapes")
    st.markdown("1) Importer vos PDF\n\n2) Cliquer **Extraction**\n\n3) Poser vos questions")

    pdf_docs = st.file_uploader("Importer des PDF", type=["pdf"], accept_multiple_files=True)

    if st.button("Extraction"):
        if not OPENAI_API_KEY:
            st.error("Ajoutez votre **OPENAI_API_KEY** dans *Settings → Secrets* ou .env.")
        elif not pdf_docs:
            st.warning("Veuillez importer au moins un PDF.")
        else:
            with st.spinner("Indexation en cours…"):
                try:
                    chunks = extract_chunks_from_pdfs(pdf_docs)
                    if not chunks:
                        st.error("Aucun texte exploitable trouvé dans les PDF.")
                    else:
                        vectorstore = create_vectorstore(chunks)
                        st.session_state.conversation = build_conversation_chain(vectorstore)
                        st.session_state.pdf_ready = True
                        st.success(f"Indexation terminée ✅ ({len(chunks)} chunks)")
                except Exception as e:
                    st.error(f"Échec de l'indexation : {e}")

# Affichage de l'historique si présent
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("Historique")
    for i, msg in enumerate(st.session_state.chat_history):
        if msg.type == "human":
            message(msg.content, is_user=True, key=f"hist_u_{i}")
        else:
            message(msg.content, key=f"hist_a_{i}")
