import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
import streamlit as st
import tempfile
import re
import requests
from dotenv import load_dotenv

import pdfplumber
import pytesseract
import cv2
import numpy as np
from PIL import Image

from langdetect import detect
from sentence_transformers import CrossEncoder

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever


from langchain_core.documents import Document
from langchain_groq import ChatGroq


# ---------------- TESSERACT ----------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ---------------- ENV ----------------
load_dotenv()


# ---------------- OPENROUTER INIT ----------------
openrouter_key = os.getenv("OPENROUTER_API_KEY")
openrouter_available = bool(openrouter_key)


# ---------------- GROQ ----------------
if "groq" not in st.session_state:
    st.session_state.groq = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )


# ---------------- MULTI LLM ----------------
def multi_llm(prompt):

    # 1️⃣ OPENROUTER FIRST
    if openrouter_available:
        try:
            st.info("🔵 Using OpenRouter...")

            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openrouter_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "openai/gpt-4o-mini",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                }
            )

            result = response.json()

            if "choices" in result:
                text = result["choices"][0]["message"]["content"]
                st.session_state["model_used"] = "OpenRouter"
                return text

        except Exception as e:
            st.warning(f"⚠️ OpenRouter failed: {e}")

    # 2️⃣ GROQ FALLBACK
    try:
        st.info("🟠 Using Groq...")
        response = st.session_state.groq.invoke(prompt)

        if response.content:
            st.session_state["model_used"] = "Groq"
            return response.content

    except Exception as e:
        st.warning(f"⚠️ Groq failed: {e}")

    st.session_state["model_used"] = "None"
    return "❌ All AI APIs failed."


# ---------------- SESSION ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "reranker" not in st.session_state:
    st.session_state.reranker = None


# ---------------- FUNCTIONS ----------------

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"


def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def preprocess_image(img):
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return Image.fromarray(thresh)

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
def process_pdf(uploaded_file):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    docs = []

    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):

            text = page.extract_text()

            if not text or len(text.strip()) < 50:
                try:
                    img = page.to_image(resolution=300).original
                    img = preprocess_image(img)
                    text = pytesseract.image_to_string(img)
                except:
                    text = ""

            text = clean_text(text)

            if len(text) > 50:
                docs.append(Document(page_content=text, metadata={"page": i+1}))

    if not docs:
        st.error("❌ No text extracted from PDF")
        return None, None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300
    )

    chunks = splitter.split_documents(docs)

    if not chunks:
        st.error("❌ Chunking failed")
        return None, None

    embeddings = load_embeddings()

    vectordb = Chroma.from_documents(chunks, embeddings)

    vector_retriever = vectordb.as_retriever(search_kwargs={"k": 60})


    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 25

   

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    return bm25, vector_retriever, reranker


def rerank(query, docs, reranker, top_k=8):
    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)
    scored = list(zip(docs, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in scored[:top_k]]


# ---------------- UI ----------------

st.title("⚖️ Legal RAG System")

file = st.file_uploader("Upload PDF", type="pdf")

if file:

    if st.session_state.retriever is None:
        with st.spinner("Processing PDF..."):
            bm25, vector_retriever, rr = process_pdf(file)

        st.session_state.bm25 = bm25
        st.session_state.vector = vector_retriever
        st.session_state.reranker = rr

        st.success("✅ PDF processed")

    q = st.chat_input("Ask your question...")

    if q:
        q = q.lower()
        original_q = q
        q = q.lower().strip()

        lang = detect_language(q)

        if lang in ["te", "hi"]:
         translated = multi_llm(f"Translate to English:\n{q}")
    if translated:
        q = translated.lower().strip()

        queries = [q]

        all_docs = []

        for query in queries:
            bm25_docs = st.session_state.bm25.invoke(q)
            vector_docs = st.session_state.vector.invoke(q)

            docs = bm25_docs + vector_docs
            docs = list({d.page_content: d for d in docs}.values())
            docs = rerank(q, docs, st.session_state.reranker, top_k=10)
            

        if docs:
            context = ""
            for d in docs:
                context += f"\n--- PAGE {d.metadata.get('page')} ---\n{d.page_content}\n"

            prompt = f"""
Answer using ONLY the given context.

If answer is partially available, answer based on available context.

If completely not found, say:
Not enough information in the document.

Context:
{context}

Question:
{q}

Answer:
"""
            ans = multi_llm(prompt)
        else:
            ans = "Not enough information"

        st.session_state.chat_history.append(("user", q))
        st.session_state.chat_history.append(("bot", ans))

    for role, msg in st.session_state.chat_history:
        st.chat_message("user" if role=="user" else "assistant").write(msg)

    if st.session_state.chat_history:
        last = st.session_state.chat_history[-1][1]

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Telugu"):
                st.write(multi_llm(f"Translate to Telugu:\n{last}"))

        with col2:
            if st.button("Hindi"):
                st.write(multi_llm(f"Translate to Hindi:\n{last}"))

    if "model_used" in st.session_state:
        st.info(f"🤖 Model Used: {st.session_state['model_used']}")