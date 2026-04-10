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
import hashlib
import pickle
from pathlib import Path

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

OCR_THRESHOLD = 100  # Min characters before OCR (increased from 50)
OCR_DPI = 200  # Reduced from 300 for faster processing
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300
TOP_K_RETRIEVAL = 8

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
            st.warning(f"⚠️ OpenRouter failed: {e}")

    if groq_available and "groq" in st.session_state:
        try:
            response = st.session_state.groq.invoke(prompt)
            if response.content:
                st.session_state["model_used"] = "Groq"
                return response.content
        except Exception as e:
            st.warning(f"⚠️ Groq failed: {e}")

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

# ============ UTILITY FUNCTIONS ============

def get_pdf_hash(file_bytes):
    """Generate hash of PDF for caching"""
    return hashlib.md5(file_bytes).hexdigest()

def save_to_cache(pdf_hash, bm25, vector, reranker):
    """Save processed PDF components to cache"""
    try:
        cache_file = CACHE_DIR / f"{pdf_hash}.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump({"bm25": bm25, "vector": vector, "reranker": reranker}, f)
    except Exception as e:
        st.warning(f"⚠️ Cache save failed: {e}")

def load_from_cache(pdf_hash):
    """Load processed PDF components from cache"""
    try:
        cache_file = CACHE_DIR / f"{pdf_hash}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
                return data["bm25"], data["vector"], data["reranker"]
    except Exception as e:
        st.warning(f"⚠️ Cache load failed: {e}")
    return None, None, None

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
    bm25, vector, reranker = load_from_cache(pdf_hash)
    
    if bm25 and vector and reranker:
        st.success("✅ Loaded from cache (faster!)")
        return bm25, vector, reranker, pdf_hash
    
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
                status_text.text(f"Processing page {i+1}/{total_pages}...")
                
                # Extract text
                text = page.extract_text() or ""
                
                # Only use OCR if text extraction was poor
                if len(text.strip()) < OCR_THRESHOLD:
                    status_text.text(f"Running OCR on page {i+1}/{total_pages}...")
                    text = extract_text_from_page_image(page, path, i+1)
                
                text = clean_text(text)
                
                if len(text) > 50:
                    docs.append(Document(page_content=text, metadata={"page": i+1}))
    
    except Exception as e:
        st.error(f"❌ PDF processing failed: {e}")
        return None, None, None, None

    if not docs:
        st.error("❌ No text extracted")
        return None, None, None, None

    # Chunking
    status_text.text("Chunking text...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    if not chunks:
        st.error("❌ Chunking failed")
        return None, None, None, None

    # Embeddings & Vector DB
    status_text.text("Creating embeddings...")
    embeddings = load_embeddings()
    vectordb = Chroma.from_documents(chunks, embeddings)
    vector_retriever = vectordb.as_retriever(search_kwargs={"k": 60})

    # BM25
    status_text.text("Creating BM25 index...")
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 25

    # Reranker
    status_text.text("Loading reranker...")
    reranker = load_reranker()

    # Clear progress
    progress_bar.empty()
    status_text.empty()

    # Save to cache
    save_to_cache(pdf_hash, bm25, vector_retriever, reranker)

    return bm25, vector_retriever, reranker, pdf_hash

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
**Tips for faster processing:**
- Use PDFs with clear text (not scanned)
- Remove unnecessary pages
- Check `Cache` folder - it reuses processed PDFs!
""")

file = st.file_uploader("Upload PDF", type="pdf")

if file:
    file_bytes = file.read()
    pdf_hash = get_pdf_hash(file_bytes)
    file.seek(0)
    
    # Check if we need to reprocess
    if st.session_state.pdf_hash != pdf_hash or st.session_state.bm25 is None:
        with st.spinner("⏳ Processing PDF (first time may take a while)..."):
            bm25, vector_retriever, reranker, hash_val = process_pdf(file)

        if bm25 is None:
            st.error("❌ PDF processing failed. Try a different file.")
        else:
            st.session_state.bm25 = bm25
            st.session_state.vector = vector_retriever
            st.session_state.reranker = reranker
            st.session_state.pdf_hash = hash_val
            st.success("✅ PDF ready!")
    else:
        st.success("✅ Using cached PDF")

    # Chat interface
    q = st.chat_input("Ask your question...")

    if q:
        q_lower = q.lower().strip()
        original_q = q

        # Language detection & translation
        lang = detect_language(q)
        translated = None

        if lang in ["te", "hi"]:
            with st.spinner("🌐 Translating..."):
                translated = multi_llm(f"Translate to English:\n{q}")

        final_query = translated.lower().strip() if translated else q_lower

        # Retrieve documents
        with st.spinner("🔍 Searching..."):
            bm25_docs = st.session_state.bm25.invoke(final_query)
            vector_docs = st.session_state.vector.invoke(final_query)
            
            # Deduplicate
            docs = bm25_docs + vector_docs
            docs = list({d.page_content: d for d in docs}.values())
            
            # Rerank
            docs = rerank(final_query, docs, st.session_state.reranker)

        # Generate answer
        if docs:
            context = "\n".join([
                f"\n--- PAGE {d.metadata.get('page', '?')} ---\n{d.page_content}"
                for d in docs
            ])

            prompt = f"""Answer using ONLY the given context.
If answer is partially available, answer based on available context.
If completely not found, say: Not enough information in the document.

Context:
{context}

Question:
{final_query}

Answer:"""

            with st.spinner("💭 Generating answer..."):
                ans = multi_llm(prompt)
        else:
            ans = "Not enough information in the document."

        # Store history
        st.session_state.chat_history.append(("user", final_query))
        st.session_state.chat_history.append(("bot", ans))

    # Display chat
    for role, msg in st.session_state.chat_history:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.write(msg)

    # Translation buttons
    if st.session_state.chat_history:
        st.divider()
        st.subheader("Translate Last Answer")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🇹🇪 Telugu"):
                last_answer = st.session_state.chat_history[-1][1]
                with st.spinner("Translating..."):
                    translated = multi_llm(f"Translate to Telugu:\n{last_answer}")
                st.write(translated)

        with col2:
            if st.button("🇮🇳 Hindi"):
                last_answer = st.session_state.chat_history[-1][1]
                with st.spinner("Translating..."):
                    translated = multi_llm(f"Translate to Hindi:\n{last_answer}")
                st.write(translated)

    # Footer
    if "model_used" in st.session_state:
        st.divider()
        st.caption(f"🤖 Model: {st.session_state['model_used']}")
