import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# LangChain (versions récentes)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from streamlit_chat import message

# ======================================================
# CONFIGURATION (clé sk-proj-… + project supportés)
# ======================================================
st.set_page_config(page_title="Chat-PDF", page_icon="📄", layout="wide")

# 1) Charger .env (local) et secrets (Cloud)
load_dotenv()

def _get_secret(name, default=None):
    # Streamlit Cloud → st.secrets ; local → .env / env
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

# Export vers l’environnement (utilisé par le SDK OpenAI sous-jacent)
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if OPENAI_PROJECT:
    os.environ["OPENAI_PROJECT"] = OPENAI_PROJECT

# Garde-fous
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY manquante. Mets-la dans `.env` (local) ou Settings → Secrets (Cloud).")
    st.stop()
if OPENAI_API_KEY.startswith("sk-proj-") and not OPENAI_PROJECT:
    st.error("Tu utilises une clé `sk-proj-…` : ajoute aussi `OPENAI_PROJECT=proj_xxx` dans `.env`/Secrets.")
    st.stop()

# ======================================================
# FONCTIONS
# ======================================================
def get_extract_chunks(pdf_docs):
    """Lit les PDFs et fait des chunks."""
    content = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            content += page.extract_text() or ""

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1400,
        chunk_overlap=320,
        length_function=len,
    )
    return splitter.split_text(content)

def create_vectorstore(chunks):
    # ✅ Utilise langchain_openai + project
    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        model=OPENAI_EMBEDDING_MODEL,
        project=OPENAI_PROJECT,  # important avec sk-proj
    )
    return FAISS.from_texts(texts=chunks, embedding=embeddings)

def get_conversation_chain(vectorstore):
    # ✅ Utilise ChatOpenAI côté langchain_openai + project
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL,
        temperature=0.2,
        project=OPENAI_PROJECT,  # important avec sk-proj
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
    )

def get_user_input(user_input):
    if not st.session_state.pdf_processed:
        st.info("Please upload PDF files")
        return
    answer = st.session_state.conversation({'question': user_input})
    st.session_state.chat_history = answer['chat_history']
    for index, value in enumerate(st.session_state.chat_history):
        if index % 2 == 0:
            message(value.content, is_user=True, key=str(index) + '_user')
        else:
            message(value.content, key=str(index) + '_assistant')

# ======================================================
# UI
# ======================================================
def main():
    st.header("Welcome to Chat PDF")
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False

    user_input = st.text_input("Please enter your question...")
    if user_input:
        get_user_input(user_input)

    with st.sidebar:
        st.subheader("Please upload PDF(s)")
        pdf_docs = st.file_uploader("Upload files", type="pdf", accept_multiple_files=True)
        if st.button("Extraction"):
            if not pdf_docs:
                st.warning("Upload at least one PDF.")
            else:
                with st.spinner("Indexing…"):
                    try:
                        chunks = get_extract_chunks(pdf_docs)
                        vectorstore = create_vectorstore(chunks)  # <-- 401 se produisait ici
                        st.session_state.pdf_processed = True
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        st.success(f"Indexation terminée ✅ ({len(chunks)} chunks)")
                    except Exception as e:
                        st.error(f"Échec de l'indexation : {e}")

    # Affichage historique
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("History")
        for i, msg in enumerate(st.session_state.chat_history):
            if getattr(msg, "type", "") == "human":
                message(msg.content, is_user=True, key=f"hist_u_{i}")
            else:
                message(msg.content, key=f"hist_a_{i}")

if __name__ == '__main__':
    main()
