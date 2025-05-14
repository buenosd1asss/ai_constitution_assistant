import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

# Путь к файлу
file_path = "constitution.txt"

# Кэшируем загрузку и векторизацию
@st.cache_resource
def load_constitution_vectorstore():
    loader = TextLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma.from_documents(chunks, embedding, persist_directory="db_constitution")

# Интерфейс
st.set_page_config(page_title="Конституционный Ассистент", layout="wide")
st.title("🧠 Конституционный AI Ассистент")

query = st.text_input("Введите ваш вопрос по Конституции:")

if query:
    with st.spinner("Обрабатываю..."):
        vectorstore = load_constitution_vectorstore()
        docs = vectorstore.similarity_search(query, k=3)

        st.subheader("🔎 Найденные фрагменты:")
        for doc in docs:
            st.markdown(f"- {doc.page_content}")

        st.success("Готово!")
