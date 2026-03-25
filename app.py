import streamlit as st
import re, os, tempfile, platform
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
import logging
import requests
import pytesseract
from PIL import Image
import fitz
import cv2
import numpy as np

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 🔥 OCR SETUP (FIXED FOR CLOUD)
try:
    if platform.system() == "Windows":
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    else:
        pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

    pytesseract.get_tesseract_version()
    OCR_AVAILABLE = True
except:
    OCR_AVAILABLE = False

# 🔥 ENV
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# 🔥 LOGGING
logging.basicConfig(filename="chatbot.log", level=logging.INFO)

# 🔥 SESSION
for k in ["chat_history","pdf_name","result"]:
    if k not in st.session_state:
        st.session_state[k] = None if k!="chat_history" else []

if "history" not in st.session_state:
    st.session_state.history = []

# 🔥 EMBEDDING
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# 🔥 TRANSLATE
def translate_to_english(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        return text

# 🔥 OCR FUNCTION (CLEAN + ACCURATE)
def ocr_extract(img):
    img_np = np.array(img)

    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

    return pytesseract.image_to_string(
        gray,
        lang='eng',
        config='--oem 3 --psm 6'
    )

# 🔥 PDF EXTRACTION (SMART LOGIC)
def extract_pdf_pages(file_path):

    doc = fitz.open(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    docs = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        # ✅ TRY NORMAL TEXT FIRST
        text = page.get_text("text")

        # 🔥 IF EMPTY → OCR
        if not text.strip() and OCR_AVAILABLE:
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            text = ocr_extract(img)

        text = re.sub(r'\s+', ' ', text).strip()

        if not text:
            continue

        chunks = splitter.split_text(text)

        for chunk in chunks:
            docs.append(Document(
                page_content=chunk,
                metadata={"page": page_num + 1}
            ))

    return docs

# 🔥 GROQ CALL
def groq_call(messages):
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "temperature": 0.2
    }

    res = requests.post(url, headers=headers, json=data)

    if res.status_code == 200:
        return res.json()["choices"][0]["message"]["content"]
    return "Error from Groq"

# 🔥 RETRIEVE (FIXED)
def retrieve(query, vectorstore):
    results = vectorstore.similarity_search(query, k=5)

    return [
        {"page": doc.metadata["page"], "text": doc.page_content}
        for doc in results
    ]

# 🔥 ASK (STRICT CONTEXT)
def ask_groq(query, pages):

    if not pages:
        return {"answer": "I could not find this in the document.", "sources": []}

    context = ""
    for p in pages:
        context += f"\n[PAGE {p['page']}]\n{p['text']}"

    response = groq_call([
        {
            "role": "system",
            "content": "Answer ONLY from the context. If not found say 'Not found in document'."
        },
        {
            "role": "user",
            "content": f"{query}\n\n{context}"
        }
    ])

    return {"answer": response, "sources": pages}

# ───── UI ─────
st.title("💬 Multilingual PDF Chatbot")

uploaded = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded:
    if uploaded.name != st.session_state.pdf_name:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        docs = extract_pdf_pages(tmp_path)

        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model
        )

        st.session_state.vectorstore = vectorstore
        st.session_state.pdf_name = uploaded.name

        st.success("✅ PDF processed!")

# 🔥 QUESTION
user_input = st.text_input("Ask in any language")

if st.button("Ask") and user_input:

    if "vectorstore" not in st.session_state:
        st.warning("Upload PDF first")
        st.stop()

    english_query = translate_to_english(user_input)

    pages = retrieve(english_query, st.session_state.vectorstore)

    result = ask_groq(english_query, pages)

    st.session_state.result = result

# 🔥 OUTPUT
result = st.session_state.get("result")

if result:
    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Sources")
    for s in result["sources"]:
        with st.expander(f"Page {s['page']}"):
            st.write(s["text"])

# 🔥 TRANSLATE OUTPUT
lang = st.selectbox("Translate to", ["English", "Telugu", "Hindi"])

if st.button("Translate") and result:
    text = result["answer"]

    if lang == "Telugu":
        text = GoogleTranslator(source='auto', target='te').translate(text)
    elif lang == "Hindi":
        text = GoogleTranslator(source='auto', target='hi').translate(text)

    st.success(text)