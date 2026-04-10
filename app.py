import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============ ROBUST NLTK SETUP ============
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.data.find('tokenizers/punkt')
except (LookupError, Exception):
    try:
        nltk.download('punkt', quiet=True)
    except Exception as e:
        pass

import streamlit as st
import tempfile
import re
import requests
from dotenv import load_dotenv
import hashlib
import pickle
from pathlib import Path
import shutil

import pdfplumber
import pytesseract
import cv2
import numpy as np
from PIL import Image

try:
    from pdf2image import convert_from_bytes
    pdf2image_available = True
except ImportError:
    convert_from_bytes = None
    pdf2image_available = False

from langdetect import detect
from sentence_transformers import CrossEncoder

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever

from langchain_core.documents import Document
from langchain_groq import ChatGroq

# ============ CONFIGURATION ============
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

# Chroma persistent directory (fix for zipfile issues)
CHROMA_DB_DIR = Path(".chroma_db")
CHROMA_DB_DIR.mkdir(exist_ok=True)

OCR_THRESHOLD = 100
OCR_DPI = 200
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300
TOP_K_RETRIEVAL = 8
BM25_K = 15  # Increased for better retrieval
VECTOR_K = 15  # Increased for better retrieval

# ============ TESSERACT ============
if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ============ ENV ============
load_dotenv()

openrouter_key = os.getenv("OPENROUTER_API_KEY")
openrouter_available = bool(openrouter_key)

groq_key = os.getenv("GROQ_API_KEY")
groq_available = bool(groq_key)

if "groq" not in st.session_state and groq_available:
    st.session_state.groq = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.2,
        groq_api_key=groq_key
    )

# ============ MULTI LLM ============
def multi_llm(prompt):
    """Call OpenRouter with Groq fallback"""
    if openrouter_available:
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openrouter_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "openai/gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=30
            )

            result = response.json()
            if "choices" in result:
                st.session_state["model_used"] = "OpenRouter"
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            pass

    if groq_available and "groq" in st.session_state:
        try:
            response = st.session_state.groq.invoke(prompt)
            if response.content:
                st.session_state["model_used"] = "Groq"
                return response.content
        except Exception as e:
            pass

    st.session_state["model_used"] = "None"
    return "❌ All AI APIs failed."

# ============ SESSION ============
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "bm25" not in st.session_state:
    st.session_state.bm25 = None
if "vector" not in st.session_state:
    st.session_state.vector = None
if "reranker" not in st.session_state:
    st.session_state.reranker = None
if "pdf_hash" not in st.session_state:
    st.session_state.pdf_hash = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []

# ============ UTILITY FUNCTIONS ============

def get_pdf_hash(file_bytes):
    """Generate hash of PDF for caching"""
    return hashlib.md5(file_bytes).hexdigest()

def save_to_cache(pdf_hash, bm25, vector, reranker, chunks):
    """Save processed PDF components to cache"""
    try:
        cache_file = CACHE_DIR / f"{pdf_hash}.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump({
                "bm25": bm25, 
                "vector": vector, 
                "reranker": reranker,
                "chunks": chunks
            }, f)
    except Exception as e:
        pass

def load_from_cache(pdf_hash):
    """Load processed PDF components from cache"""
    try:
        cache_file = CACHE_DIR / f"{pdf_hash}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
                return data["bm25"], data["vector"], data["reranker"], data.get("chunks", [])
    except Exception as e:
        pass
    return None, None, None, []

def detect_language(text):
    """Detect text language"""
    try:
        return detect(text)
    except:
        return "en"

def clean_text(text):
    """Clean extracted text"""
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess_image(img):
    """Preprocess image for OCR"""
    img = np.array(img)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return Image.fromarray(thresh)

def extract_text_from_page_image(page, pdf_path=None, page_num=None):
    """Extract text from page using OCR with fallback"""
    try:
        img = page.to_image(resolution=OCR_DPI).original
        img = preprocess_image(img)
        return pytesseract.image_to_string(img)
    except Exception as e:
        if pdf2image_available and pdf_path:
            try:
                with open(pdf_path, "rb") as f:
                    images = convert_from_bytes(
                        f.read(), dpi=OCR_DPI,
                        first_page=page_num or page.page_number,
                        last_page=page_num or page.page_number
                    )
                img = preprocess_image(images[0])
                return pytesseract.image_to_string(img)
            except Exception as ocr_error:
                pass
        return ""

@st.cache_resource
def load_embeddings():
    """Load embedding model (cached)"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_reranker():
    """Load reranker model (cached)"""
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def process_pdf(uploaded_file):
    """Process PDF with progress tracking and OCR skipping for text-heavy PDFs"""
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    
    # Check cache first
    pdf_hash = get_pdf_hash(file_bytes)
    bm25, vector, reranker, cached_chunks = load_from_cache(pdf_hash)
    
    if bm25 and vector and reranker:
        st.success("✅ Loaded from cache (instant!)")
        return bm25, vector, reranker, pdf_hash, cached_chunks
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        path = tmp.name

    docs = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        with pdfplumber.open(path) as pdf:
            total_pages = len(pdf.pages)
            
            for i, page in enumerate(pdf.pages):
                # Update progress
                progress = (i + 1) / total_pages
                progress_bar.progress(progress)
                status_text.text(f"📄 Processing page {i+1}/{total_pages}...")
                
                # Extract text
                text = page.extract_text() or ""
                
                # Only use OCR if text extraction was poor
                if len(text.strip()) < OCR_THRESHOLD:
                    status_text.text(f"🔍 OCR page {i+1}/{total_pages}...")
                    text = extract_text_from_page_image(page, path, i+1)
                
                text = clean_text(text)
                
                if len(text) > 50:
                    docs.append(Document(page_content=text, metadata={"page": i+1}))
    
    except Exception as e:
        st.error(f"❌ PDF processing failed: {e}")
        return None, None, None, None, []

    if not docs:
        st.error("❌ No text extracted")
        return None, None, None, None, []

    # Chunking
    status_text.text("✂️ Chunking text...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    if not chunks:
        st.error("❌ Chunking failed")
        return None, None, None, None, []

    # Embeddings & Vector DB
    status_text.text("🧠 Creating embeddings...")
    embeddings = load_embeddings()
    
    # Clean up old Chroma DB to avoid zipfile issues
    chroma_path = CHROMA_DB_DIR / f"chroma_{pdf_hash}"
    if chroma_path.exists():
        try:
            shutil.rmtree(chroma_path)
        except:
            pass
    
    vectordb = Chroma.from_documents(
        chunks, 
        embeddings,
        persist_directory=str(chroma_path)
    )
    vector_retriever = vectordb.as_retriever(search_kwargs={"k": VECTOR_K})

    # BM25
    status_text.text("📑 Creating BM25 index...")
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = BM25_K

    # Reranker
    status_text.text("⚖️ Loading reranker...")
    reranker = load_reranker()

    # Clear progress
    progress_bar.empty()
    status_text.empty()

    # Save to cache
    save_to_cache(pdf_hash, bm25, vector_retriever, reranker, chunks)

    return bm25, vector_retriever, reranker, pdf_hash, chunks

def rerank(query, docs, reranker, top_k=TOP_K_RETRIEVAL):
    """Rerank documents using cross-encoder"""
    if not docs:
        return []
    
    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)
    scored = list(zip(docs, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in scored[:top_k]]

# ============ UI ============

st.set_page_config(page_title="⚖️ Legal RAG System", layout="wide")
st.title("⚖️ Legal RAG System")

st.markdown("""
**💡 Tips for best results:**
- Use PDFs with clear text (not scanned)
- Ask specific questions
- Check `.cache` folder for instant reloads!
""")

file = st.file_uploader("Upload PDF", type="pdf")

if file:
    file_bytes = file.read()
    pdf_hash = get_pdf_hash(file_bytes)
    file.seek(0)
    
    # Check if we need to reprocess
    if st.session_state.pdf_hash != pdf_hash or st.session_state.bm25 is None:
        with st.spinner("⏳ Processing PDF..."):
            bm25, vector_retriever, reranker, hash_val, chunks = process_pdf(file)

        if bm25 is None:
            st.error("❌ PDF processing failed. Try another file.")
        else:
            st.session_state.bm25 = bm25
            st.session_state.vector = vector_retriever
            st.session_state.reranker = reranker
            st.session_state.pdf_hash = hash_val
            st.session_state.chunks = chunks
            st.success("✅ PDF ready!")
    else:
        st.success("✅ Using cached PDF")

    # Chat interface
    q = st.chat_input("Ask your question...")

    if q:
        q_original = q
        q_lower = q.lower().strip()

        # Language detection & translation
        lang = detect_language(q)
        final_query = q_lower
        translated_to_en = None

        if lang in ["te", "hi", "ta", "ml", "kn"]:
            with st.spinner("🌐 Translating to English..."):
                translated_to_en = multi_llm(f"Translate this {lang} text to English. Only return the translation, nothing else:\n{q}")
                if translated_to_en and len(translated_to_en) > 0:
                    final_query = translated_to_en.lower().strip()

        # Retrieve documents
        with st.spinner("🔍 Searching document..."):
            try:
                bm25_docs = st.session_state.bm25.invoke(final_query)
                vector_docs = st.session_state.vector.invoke(final_query)
                
                # Combine and deduplicate
                docs = bm25_docs + vector_docs
                unique_docs = {}
                for d in docs:
                    key = d.page_content[:100]  # Use first 100 chars as key
                    if key not in unique_docs:
                        unique_docs[key] = d
                docs = list(unique_docs.values())
                
                # Rerank
                docs = rerank(final_query, docs, st.session_state.reranker, top_k=TOP_K_RETRIEVAL)
            except Exception as e:
                st.error(f"Retrieval error: {e}")
                docs = []

        # Generate answer
        if docs:
            context = "\n".join([
                f"\n--- PAGE {d.metadata.get('page', '?')} ---\n{d.page_content}"
                for d in docs
            ])

            prompt = f"""Answer the question using ONLY the given context. Be concise and direct.

If you find relevant information, provide it.
If information is not in the context, say: 'The document does not contain information about this topic.'

Context:
{context}

Question: {final_query}

Answer:"""

            with st.spinner("💭 Generating answer..."):
                ans = multi_llm(prompt)
        else:
            ans = "The document does not contain information about this topic."

        # Store history - Don't show translation as error
        st.session_state.chat_history.append(("user", q_original))
        st.session_state.chat_history.append(("bot", ans))

    # Display chat history
    for role, msg in st.session_state.chat_history:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.write(msg)

    # Translation buttons
    if st.session_state.chat_history:
        st.divider()
        st.subheader("🌍 Translate Last Answer")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🇹🇪 Telugu"):
                last_answer = st.session_state.chat_history[-1][1]
                with st.spinner("Translating..."):
                    translated = multi_llm(f"Translate to Telugu. Return only the translation:\n{last_answer}")
                    st.info(translated)

        with col2:
            if st.button("🇮🇳 Hindi"):
                last_answer = st.session_state.chat_history[-1][1]
                with st.spinner("Translating..."):
                    translated = multi_llm(f"Translate to Hindi. Return only the translation:\n{last_answer}")
                    st.info(translated)

        with col3:
            if st.button("🔄 Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()

    # Footer
    if "model_used" in st.session_state:
        st.divider()
        st.caption(f"🤖 Model: {st.session_state['model_used']}")
