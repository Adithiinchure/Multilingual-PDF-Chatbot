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
groq_key = os.getenv("GROQ_API_KEY")
if groq_key and "groq" not in st.session_state:
    st.session_state.groq = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0,
        api_key=groq_key
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
                        {"role": "system", "content": "You are a legal assistant. Answer based only on provided context."},
                        {"role": "user", "content": prompt}
                    ]
                },
                timeout=30
            )

            result = response.json()

            if "choices" in result and result.get("choices"):
                text = result["choices"][0]["message"]["content"]
                st.session_state["model_used"] = "OpenRouter (GPT-4o Mini)"
                return text
            else:
                st.warning(f"⚠️ OpenRouter error: {result.get('error', 'Unknown error')}")

        except Exception as e:
            st.warning(f"⚠️ OpenRouter failed: {str(e)}")

    # 2️⃣ GROQ FALLBACK
    if groq_key:
        try:
            st.info("🟠 Using Groq...")
            response = st.session_state.groq.invoke(prompt)

            if response.content:
                st.session_state["model_used"] = "Groq (Llama 3.3 70B)"
                return response.content
            else:
                st.warning("⚠️ Groq returned empty response")

        except Exception as e:
            st.warning(f"⚠️ Groq failed: {str(e)}")
    else:
        st.warning("⚠️ GROQ_API_KEY not set in .env")

    st.session_state["model_used"] = "None"
    return "❌ All AI APIs failed. Check your API keys (.env file)"


# ---------------- SESSION ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# FIX #1: Changed from "retriever" to "bm25" to match actual usage
if "bm25" not in st.session_state:
    st.session_state.bm25 = None

if "vector" not in st.session_state:
    st.session_state.vector = None

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
                except Exception as e:
                    st.warning(f"OCR failed for page {i+1}: {str(e)}")
                    text = ""

            text = clean_text(text)

            if len(text) > 50:
                docs.append(Document(page_content=text, metadata={"page": i+1}))

    if not docs:
        st.error("❌ No text extracted from PDF. Check if PDF is valid and readable.")
        return None, None, None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300
    )

    chunks = splitter.split_documents(docs)

    if not chunks:
        st.error("❌ Chunking failed")
        return None, None, None

    embeddings = load_embeddings()

    vectordb = Chroma.from_documents(chunks, embeddings)

    vector_retriever = vectordb.as_retriever(search_kwargs={"k": 40})

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 25

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    return bm25, vector_retriever, reranker


def rerank(query, docs, reranker, top_k=8):
    if not docs:
        return []
    
    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)
    scored = list(zip(docs, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in scored[:top_k]]


# ---------------- UI ----------------

st.title("⚖️ Legal RAG System")

st.markdown("### Upload PDF")
file = st.file_uploader("Upload PDF", type="pdf")

if file:

    # FIX #2: Check the actual variables that store retrievers
    if st.session_state.bm25 is None:
        with st.spinner("🔄 Processing PDF... This may take a minute."):
            bm25, vector_retriever, rr = process_pdf(file)

        if bm25 is not None:
            st.session_state.bm25 = bm25
            st.session_state.vector = vector_retriever
            st.session_state.reranker = rr
            st.success("✅ PDF processed successfully!")
        else:
            st.error("Failed to process PDF")
            st.stop()

    q = st.chat_input("Ask your question...")

    if q:

        lang = detect_language(q)

        # Translate if needed
        if lang == "te":
            st.info("🔄 Translating Telugu to English...")
            q = multi_llm(f"Translate this Telugu text to English. Return ONLY the English translation:\n{q}")
        elif lang == "hi":
            st.info("🔄 Translating Hindi to English...")
            q = multi_llm(f"Translate this Hindi text to English. Return ONLY the English translation:\n{q}")

        st.info("🔍 Generating search queries...")
        queries_text = multi_llm(f"Generate 5 alternative search queries for this question. Return only the queries, one per line:\n{q}")
        queries = [x.strip() for x in queries_text.split("\n") if x.strip()]
        queries.append(q)

        all_docs = []

        st.info(f"📚 Searching with {len(queries)} queries...")
        for query in queries:
            try:
                bm25_docs = st.session_state.bm25.invoke(query)
                vector_docs = st.session_state.vector.invoke(query)
                all_docs.extend(bm25_docs + vector_docs)
            except Exception as e:
                st.warning(f"Search error: {str(e)}")

        # Remove duplicates
        docs = list({d.page_content: d for d in all_docs}.values())
        
        # Rerank results
        docs = rerank(q, docs, st.session_state.reranker, top_k=8)

        if docs:
            st.info(f"📄 Found {len(docs)} relevant sections")
            context = ""
            for d in docs:
                context += f"\n--- PAGE {d.metadata.get('page', 'N/A')} ---\n{d.page_content}\n"

            prompt = f"""You are a legal assistant. Answer ONLY based on the provided context.
If information is not in the context, say "The provided context does not contain information about this."

Context:
{context}

Question:
{q}

Answer (cite page numbers):"""
            
            ans = multi_llm(prompt)
        else:
            ans = "❌ No relevant information found in the PDF. Try a different question."

        st.session_state.chat_history.append(("user", q))
        st.session_state.chat_history.append(("bot", ans))

    # Display chat history
    st.markdown("### Conversation")
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.chat_message("user").write(msg)
        else:
            st.chat_message("assistant").write(msg)

    # Translation buttons
    if st.session_state.chat_history:
        last = st.session_state.chat_history[-1][1]

        st.markdown("### Translate Last Answer")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📍 Telugu"):
                with st.spinner("Translating..."):
                    telugu = multi_llm(f"Translate to Telugu. Return ONLY the Telugu translation:\n{last}")
                    st.write(telugu)

        with col2:
            if st.button("📍 Hindi"):
                with st.spinner("Translating..."):
                    hindi = multi_llm(f"Translate to Hindi. Return ONLY the Hindi translation:\n{last}")
                    st.write(hindi)

    # Show model info
    if "model_used" in st.session_state:
        st.sidebar.info(f"🤖 Model: {st.session_state['model_used']}")

else:
    st.info("👆 Upload a PDF file to get started")