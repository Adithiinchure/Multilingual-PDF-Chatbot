import streamlit as st
from groq import Groq
import re, os
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
import logging
import requests
import tempfile
try:
    import pytesseract
    OCR_AVAILABLE = True
except:
    OCR_AVAILABLE = False
from PIL import Image
import fitz
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 🔥 OCR setup

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

logging.basicConfig(
    filename="chatbot.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ✅ SESSION
for k,v in [("chat_history",[]),("pdf_name",""),("result",None)]:
    if k not in st.session_state:
        st.session_state[k]=v

if "translated_text" not in st.session_state:
    st.session_state.translated_text = ""

if "history" not in st.session_state:
    st.session_state.history = []

st.set_page_config(page_title="Multilingual PDF Chatbot", layout="wide")

# ✅ EMBEDDINGS
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 🔥 TRANSLATE
def translate_to_english(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        return text

# ✅ PDF FUNCTION (FIXED)
def extract_pdf_pages(file_path, mode="Normal (text pdf)"):

    doc = fitz.open(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    docs = []

    for page_num in range(len(doc)):

        page = doc[page_num]

        if mode == "Normal (Fast)":
            text = page.get_text("text")
            print(f"⚡ Page {page_num+1}: Normal extraction")

        else:
            if OCR_AVAILABLE:
              print(f"🌍 Page {page_num+1}: OCR extraction")

        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        text = pytesseract.image_to_string(
            img,
            lang='eng+tel+hin'
        )
    else:
        text = ""

        text = re.sub(r'\s+', ' ', text).strip()

        chunks = splitter.split_text(text)

        for chunk in chunks:
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={"page": page_num + 1}
                )
            )

    return docs

# ✅ GROQ + OPENROUTER
def groq_call(messages):
    client = Groq(api_key=GROQ_API_KEY)

    try:
        r = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=500,
            temperature=0.2
        )
        return r.choices[0].message.content.strip()

    except:
        openrouter_response = openrouter_call(messages)
        if "⚠️" not in openrouter_response:
            return openrouter_response
        return "⚠️ All AI services failed."

def openrouter_call(messages):

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": messages
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return "⚠️ OpenRouter failed"
    except:
        return "⚠️ OpenRouter error"

# ✅ RETRIEVE
def retrieve(query, vectorstore, k=5):

    results = vectorstore.similarity_search(query, k=k*2)

    pages = []
    seen = set()

    for doc in results:
        page = doc.metadata["page"]

        if page not in seen:
            seen.add(page)
            pages.append({
                "page": page,
                "text": doc.page_content
            })

    return pages

# ✅ ASK
def ask_groq(query, pages):

    if not pages:
        return {"answer": "Answer not found", "sources": []}

    context = ""
    for p in pages:
        context += f"\n\n[PAGE {p['page']}]\n{p['text']}"

    raw = groq_call([
        {"role": "system", "content": "Answer from context only"},
        {"role": "user", "content": f"{query}\n\n{context}"}
    ])

    return {"answer": raw, "sources": pages}

# ───────── SIDEBAR ─────────
# ───────── SIDEBAR ─────────
with st.sidebar:

    mode = st.radio("Mode", ["Normal (text pdf english)", "OCR (Multilingual)"])

    uploaded = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded:
        if uploaded.name != st.session_state.pdf_name:

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            docs = extract_pdf_pages(tmp_path, mode)

            vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=embedding_model,
                persist_directory=f"chroma_db_{uploaded.name}"
            )

            st.session_state.vectorstore = vectorstore
            st.session_state.pdf_name = uploaded.name

            st.success("✅ PDF processed!")

    # 🔥 HISTORY BELOW PDF (THIS IS YOUR REQUIRED PART)
    st.markdown("---")
    st.markdown("### 📜 History")

    for i, item in enumerate(st.session_state.history):
        if st.button(item["q"], key=f"side{i}"):
            st.session_state.selected_q = item["q"]
            st.session_state.selected_a = item["a"]


# ───────── MAIN ─────────
st.title("💬 Multilingual PDF Chatbot")

user_input = st.text_input("Ask something")

if st.button("Ask") and user_input:

    if "vectorstore" not in st.session_state:
        st.warning("Upload PDF first")
        st.stop()

    english_query = translate_to_english(user_input)

    top_pages = retrieve(english_query, st.session_state.vectorstore)

    result = ask_groq(english_query, top_pages)

    if isinstance(result, dict):

        st.session_state.result = result

        # 🔥 Save history
        st.session_state.history.insert(0, {
            "q": user_input,
            "a": result["answer"]
        })

        st.session_state.history = st.session_state.history[:10]



# ───────── OUTPUT ─────────
result = st.session_state.get("result")

if result:

    st.markdown("### 💬 Answer")
    st.write(result["answer"])

    st.markdown("### 📚 Sources")

    for s in result["sources"]:
        with st.expander(f"📄 Page {s['page']}"):
            st.write(s["text"])
if "selected_q" in st.session_state:

    st.markdown("### 📌 Selected Question")
    st.write(st.session_state.selected_q)

    st.markdown("### 💡 Answer")
    st.write(st.session_state.selected_a)

# ───────── HISTORY ─────────

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