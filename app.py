import streamlit as st
import re, os, tempfile
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
import logging
import requests
from PIL import Image
import fitz
import numpy as np
import easyocr  # ✅ NEW

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
HF_API_KEY = os.getenv("HF_API_KEY", "")
# 🔥 EasyOCR INIT (Multilingual)
reader_te = easyocr.Reader(['te','en'])
reader_hi = easyocr.Reader(['hi','en'])
reader_en = easyocr.Reader(['en'])

# 🔥 ENV
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# 🔥 LOGGING
logging.basicConfig(filename="chatbot.log", level=logging.INFO)

# 🔥 SESSION
for k,v in [("chat_history",[]),("pdf_name",""),("result",None)]:
    if k not in st.session_state:
        st.session_state[k]=v

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

# 🔥 EASY OCR FUNCTION
def ocr_extract(img):
    import numpy as np

    img_np = np.array(img)

    text_en = " ".join([r[1] for r in reader_en.readtext(img_np)])
    text_te = " ".join([r[1] for r in reader_te.readtext(img_np)])
    text_hi = " ".join([r[1] for r in reader_hi.readtext(img_np)])

    # 🔥 combine all results
    text = text_en + " " + text_te + " " + text_hi

    return text.strip()

# 🔥 PDF EXTRACTION
def extract_pdf_pages(file_path):

    doc = fitz.open(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    docs = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        # ✅ TRY NORMAL TEXT FIRST
        text = page.get_text("text")

        # 🔥 IF EMPTY → OCR
        if not text.strip():
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
        "temperature": 0.3
    }

    try:
        res = requests.post(url, headers=headers, json=data)

        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"]

        # 🔥 fallback → OpenRouter
        open_res = openrouter_call(messages)
        if open_res:
            return open_res

        # 🔥 fallback → HuggingFace
        hf_res = huggingface_call(messages)
        if hf_res:
            return hf_res

        return "⚠️ All APIs failed"

    except:
        open_res = openrouter_call(messages)
        if open_res:
            return open_res

        hf_res = huggingface_call(messages)
        if hf_res:
            return hf_res

        return "⚠️ All APIs failed"


def openrouter_call(messages):
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": messages
    }

    try:
        res = requests.post(url, headers=headers, json=data)
        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"]
        else:
            return None
    except:
        return None
    
def huggingface_call(messages):
    url = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}"
    }

    prompt = ""
    for m in messages:
        prompt += f"{m['role']}: {m['content']}\n"

    try:
        res = requests.post(url, headers=headers, json={"inputs": prompt})

        if res.status_code == 200:
            return res.json()[0]["generated_text"]
        else:
            return None
    except:
        return None
# 🔥 RETRIEVE (IMPROVED)
def retrieve(query, vectorstore):
    results = vectorstore.similarity_search(query, k=5)

    return [
        {"page": doc.metadata["page"], "text": doc.page_content}
        for doc in results
    ]

# 🔥 ASK (SMART ANSWER)
def ask_groq(query, pages):

    if not pages:
        return {"answer": "No relevant info found.", "sources": []}

    context = ""
    for p in pages:
        context += f"\n[PAGE {p['page']}]\n{p['text']}"

    response = groq_call([
        {
            "role": "system",
            "content": "Answer clearly using the context. If partial info exists, still answer helpfully."
        },
        {
            "role": "user",
            "content": f"{query}\n\n{context}"
        }
    ])

    return {"answer": response, "sources": pages}

# ───────── SIDEBAR ─────────
with st.sidebar:

    st.markdown("### Mode")
    st.radio("Mode", ["Normal (text pdf english)", "OCR (Multilingual)"])

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

    st.markdown("### History")

    for i, item in enumerate(st.session_state.history):
        if st.button(item["q"], key=f"h{i}"):
            st.session_state.result = {"answer": item["a"], "sources": []}

# ───────── MAIN ─────────
st.title("💬 Multilingual PDF Chatbot")

user_input = st.text_input("Ask something")

if st.button("Ask") and user_input:

    if "vectorstore" not in st.session_state:
        st.warning("Upload PDF first")
        st.stop()

    english_query = translate_to_english(user_input)

    pages = retrieve(english_query, st.session_state.vectorstore)

    result = ask_groq(english_query, pages)

    st.session_state.result = result

    st.session_state.history.insert(0, {
        "q": user_input,
        "a": result["answer"]
    })

# ───────── OUTPUT ─────────
result = st.session_state.get("result")

if result:
    st.markdown("### 💬 Answer")
    st.write(result["answer"])

    st.markdown("### 📚 Sources")

    for s in result["sources"]:
        with st.expander(f"Page {s['page']}"):
            st.write(s["text"])

# ───────── TRANSLATE ─────────
st.markdown("### 🌐 Translate")

lang = st.selectbox("Language", ["English", "Telugu", "Hindi"])

if st.button("Translate") and result:
    text = result["answer"]

    if lang == "Telugu":
        text = GoogleTranslator(source='auto', target='te').translate(text)
    elif lang == "Hindi":
        text = GoogleTranslator(source='auto', target='hi').translate(text)

    st.success(text)