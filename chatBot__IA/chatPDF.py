import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from streamlit_chat import message
from langchain.vectorstores import FAISS

#pip install pyPDF2 langchain python-dotenv

st.set_page_config(page_title="Chat-PDF",page_icon="gear",layout="wide")

def get_extract_chunks(pdf_docs):
    content = "" 

    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            content += page.extract_text()
    #End Extraction
    #Start chunks breakdown
    splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1400,
        chunk_overlap = 320,
        length_function=len
    )
    chunks = splitter.split_text(content)
    return chunks 

def create_vectorstore(chunks):
    Currentembedding = OpenAIEmbeddings()
    vectorestore = FAISS.from_texts(texts= chunks,embedding=Currentembedding)
    return vectorestore

def get_conversation_chain(vectorestore):
    Currentllm = ChatOpenAI()
    current_memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm= Currentllm,
        retriever = vectorestore.as_retriever(),
        memory=current_memory
    )
    return conversation_chain

def get_user_input(user_input):
    if not st.session_state.pdf_processed:
        st.info('Please upload PDF files')
        return
    answer = st.session_state.conversation({'question':user_input})
    st.session_state.chat_history = answer['chat_history']
    for index, value in enumerate(st.session_state.chat_history):
        if index % 2 == 0:
            #st.session_state.chat_history[index]
            message(value.content, is_user = True, key=str(index) + '_user')
        else:
            message(value.content, key=str(index)+ '_user')

def main():
    load_dotenv()
    st.header("Welcome to Chat PDF ADD")
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history'not in st.session_state:
        st.session_state.chat_history = None

    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    user_input = st.text_input("Please enter your question...")
    if user_input:
        get_user_input(user_input)
    with st.sidebar:
        st.subheader("Please upload PDFS")
        pdf_docs = st.file_uploader("Upload files", type="pdf", accept_multiple_files=True)
        bt = st.button("Extraction")
        okay = False
        if bt:
            with st.spinner("Encore un peu de patience..."):
                chunks = get_extract_chunks(pdf_docs)
                okay = True
                #create vectoStore
                vectoreStore = create_vectorstore(chunks)
                st.session_state.pdf_processed = True
                st.session_state.conversation = get_conversation_chain(vectoreStore)
    #Affichage text    
    #if okay:
    #    for chunk in chunks :
    #        st.success(chunk)





if __name__ == '__main__':
    main()