import streamlit as st
from groq import Groq
import re, os
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
import logging
import requests
import tempfile
import shutil

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader   # ✅ NEW

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

# ✅ NEW PDF LOADER (FIXED)
def extract_pdf_pages(file_path):

    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    docs = []

    for doc in documents:
        chunks = splitter.split_text(doc.page_content)

        for chunk in chunks:
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={"page": doc.metadata.get("page", 0) + 1}
                )
            )

    return docs

# ✅ GROQ
def groq_call(messages):
    client = Groq(api_key=GROQ_API_KEY)

    try:
        r = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=500,
            temperature=0
        )
        return r.choices[0].message.content.strip()

    except Exception as e:
        return f"⚠️ ERROR: {e}"

# ✅ FIXED RETRIEVE
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

# ✅ LLM
def ask_groq(query, pages):

    if not pages:
        return {"answer": "Answer not found in document", "sources": []}

    context = ""
    page_numbers = []

    for p in pages:
        context += f"\n\n[PAGE {p['page']}]\n{p['text']}"
        page_numbers.append(str(p["page"]))

    system = f"""
You are a document QA assistant.

RULES:
- Answer ONLY from the provided context
- ALWAYS answer in ENGLISH
- If answer is present → give exact answer
- If not → say "Answer not found in document"
"""

    user_prompt = f"""
Question: {query}

Context:
{context}

Answer:
"""

    raw = groq_call([
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt}
    ])

    return {"answer": raw, "sources": pages}

# ───────── SIDEBAR ─────────
with st.sidebar:

    uploaded = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded:
        if uploaded.name != st.session_state.pdf_name:

            # ✅ SAVE TEMP FILE
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            docs = extract_pdf_pages(tmp_path)

            vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=embedding_model,
                persist_directory=f"chroma_db_{uploaded.name}"   # ✅ FIX
            )

            vectorstore.persist()

            st.session_state.vectorstore = vectorstore
            st.session_state.pdf_name = uploaded.name
            st.session_state.chat_history = []

            st.success("✅ PDF processed!")

# ───────── MAIN ─────────
st.title("💬 Multilingual PDF Chatbot")

user_input = st.text_input("Ask something")

if  st.button("Ask") and user_input:

    if "vectorstore" not in st.session_state:
        st.warning("Upload PDF first")
        st.stop()

    english_query = translate_to_english(user_input)

    top_pages = retrieve(english_query, st.session_state.vectorstore, k=5)

    result = ask_groq(english_query, top_pages)

    if isinstance(result, dict):
        st.session_state.result = result
    else:
        st.session_state.result = {"answer": "Error occurred", "sources": []}

# ───────── OUTPUT ─────────
# ───────── OUTPUT ─────────
result = st.session_state.get("result")

if result and isinstance(result, dict):

    st.markdown("### 💬 Answer")

    answer_text = result.get("answer", "")
    answer_text = re.sub(r'\[SOURCE:.*?\]', '', answer_text).strip()

    st.write(answer_text)

    st.markdown("### 📚 Sources")

    sources = result.get("sources", [])

    shown_pages = set()

    for s in sources:
        if not isinstance(s, dict):
            continue

        page = s.get("page")

        if page and page not in shown_pages:
            shown_pages.add(page)

            with st.expander(f"📄 Page {page}", expanded=False):
                st.markdown(f"**Content from Page {page}:**")
                st.write(s.get("text", ""))

else:
    st.info("👉 Ask a question to see results")

# ───────── TRANSLATE OUTPUT ─────────
st.markdown("### 🌐 Translate Answer")

lang_choice = st.selectbox("Select Language", ["English", "Telugu", "Hindi"])

if st.button("Translate") and st.session_state.result:

    clean = re.sub(r'\[SOURCE:.*?\]', '', st.session_state.result["answer"]).strip()

    try:
        if lang_choice == "English":
            st.session_state.translated_text = clean
        elif lang_choice == "Telugu":
            st.session_state.translated_text = GoogleTranslator(source='auto', target='te').translate(clean)
        elif lang_choice == "Hindi":
            st.session_state.translated_text = GoogleTranslator(source='auto', target='hi').translate(clean)
    except:
        st.session_state.translated_text = "⚠️ Translation failed"

if st.session_state.translated_text:
    st.success(st.session_state.translated_text)