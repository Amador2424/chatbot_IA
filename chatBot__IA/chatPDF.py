import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# ✅ NOUVEAUX imports (utiliser langchain_openai, pas langchain_community)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from streamlit_chat import message

st.set_page_config(page_title="Chat-PDFAmad", page_icon="📄", layout="wide")

# ------- config clés -------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_PROJECT = os.getenv("OPENAI_PROJECT")  # requis si la clé commence par sk-proj-
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Propager aux libs sous-jacentes
if OPENAI_API_KEY: os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if OPENAI_PROJECT: os.environ["OPENAI_PROJECT"] = OPENAI_PROJECT

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY manquante.")
    st.stop()
if OPENAI_API_KEY.startswith("sk-proj-") and not OPENAI_PROJECT:
    st.error("Tu utilises une clé sk-proj-… : ajoute OPENAI_PROJECT=proj_xxx dans ton .env / Secrets.")
    st.stop()

# ------- tes fonctions -------
def get_extract_chunks(pdf_docs):
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
    # ✅ Utilise le client OpenAI récent avec support du project
    embedding = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        model=OPENAI_EMBEDDING_MODEL,
        project=OPENAI_PROJECT,  # important avec sk-proj-…
    )
    return FAISS.from_texts(texts=chunks, embedding=embedding)

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL,
        temperature=0.2,
        project=OPENAI_PROJECT,  # important avec sk-proj-…
    )
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory
    )
