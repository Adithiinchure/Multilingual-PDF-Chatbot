import streamlit as st
from groq import Groq
import re, os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np
from deep_translator import GoogleTranslator
import logging
import requests
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY", "")
logging.basicConfig(
    filename="chatbot.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Monitoring
if "fallback_count" not in st.session_state:
    st.session_state.fallback_count = 0

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

import fitz

st.set_page_config(page_title="Multilingual RAG Chatbot", page_icon="💬", layout="wide")

# Session state
for k,v in [("pdf_pages",[]),("chat_history",[]),("pdf_name","")]:
    if k not in st.session_state:
        st.session_state[k]=v

embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")

# ───────── GROQ ─────────
def groq_call(messages):
    client = Groq(api_key=GROQ_API_KEY)

    try:
        logging.info("Calling Groq API")

        r = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=800,
            temperature=0
        )

        return r.choices[0].message.content.strip()

    except Exception as e:
        error_msg = str(e)
        logging.error(f"Groq failed: {error_msg}")

        # 🔥 Fallback only on limit
        if "429" in error_msg or "rate limit" in error_msg.lower():
            logging.info("Switching to Hugging Face fallback")

            st.session_state.fallback_count += 1

            # Convert messages → single prompt
            prompt = " ".join([m["content"] for m in messages])

            return huggingface_api(prompt)

        return f"⚠️ ERROR: {e}"



def huggingface_api(prompt):
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct"

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}"
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.7
        }
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)

        if response.status_code == 200:
            return response.json()[0]["generated_text"]

        elif response.status_code == 429:
            logging.error("Hugging Face rate limit reached")
            return "⚠️ Both APIs are busy. Please try again later."

        else:
            logging.error(f"HF Error: {response.text}")
            return "⚠️ Fallback model failed."

    except Exception as e:
        logging.error(f"HF Exception: {e}")
        return "⚠️ Hugging Face error."
# ───────── LANGUAGE ─────────
def detect_language(text):
    if re.search(r'[\u0C00-\u0C7F]', text):
        return "Telugu"
    if re.search(r'[\u0900-\u097F]', text):
        return "Hindi"
    return "English"

# ───────── PDF ─────────
def extract_pdf_pages(f):
    pages = []
    doc = fitz.open(stream=f.read(), filetype="pdf")

    for i,page in enumerate(doc, start=1):
        text = re.sub(r'\s+', ' ', page.get_text())
        pages.append({"page": i, "text": text})

    return pages

# ───────── RETRIEVE ─────────
def retrieve(query, pages, top_k=3):

    query_vec = embedding_model.encode(query)

    scored = []

    for p in pages:
        chunk = p["text"][:800]
        vec = embedding_model.encode(chunk)

        sim = np.dot(query_vec, vec) / (
            np.linalg.norm(query_vec) * np.linalg.norm(vec)
        )

        scored.append({**p, "score": float(sim)})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

# ───────── LLM ─────────
def ask_groq(query, pages, lang):

    context = ""
    for p in pages:
        context += f"\n\n[PAGE {p['page']}]\n{p['text'][:800]}"

    system = """
You are a helpful and friendly assistant. Rules: - Answer in simple, natural English (like talking to a person) - Understand Telugu/Hindi but reply clearly in English - If exact answer is not found, still explain helpfully - Do NOT say NOT_FOUND directly - Keep answer short and clear - Use ONLY the provided pages - End with [SOURCE: Page X] If the question is broad (like summary): - Cover ALL topics from the document - Do NOT focus on only one topic - Give balanced summary of all sections
"""

    raw = groq_call([
        {"role":"system","content":system},
        {"role":"user","content":f"{query}\n\n{context}"}
    ])

    return {"answer": raw, "sources": pages}

# ───────── SIDEBAR ─────────
with st.sidebar:

    uploaded = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded:
        if uploaded.name != st.session_state.pdf_name:
            pages = extract_pdf_pages(uploaded)
            st.session_state.pdf_pages = pages
            st.session_state.pdf_name = uploaded.name
            st.session_state.chat_history = []
            st.success(f"{len(pages)} pages loaded")

    st.markdown("### 🕑 Chat History")

    for chat in st.session_state.chat_history:
        st.markdown(f"**Q:** {chat['question']}")
        st.markdown(f"**A:** {chat['answer'][:100]}...")

        for s in chat["source"]:
            st.caption(f"📄 Page {s['page']} - {s['text']}")

        st.markdown("---")

# ───────── MAIN ─────────
st.title("💬 Multilingual PDF Chatbot")

user_input = st.text_input("Ask something")
ask_button = st.button("Ask")

if ask_button and user_input:

    if not st.session_state.pdf_pages:
        st.warning("Upload PDF first")
        st.stop()

    lang = detect_language(user_input)
    query_len = len(user_input.split())

    # ✅ SUMMARY FIX
    if query_len > 8:
        top_pages = st.session_state.pdf_pages
    else:
        top_pages = retrieve(user_input, st.session_state.pdf_pages)

    result = ask_groq(user_input, top_pages, lang)
    st.session_state.result = result

    # ✅ SHOW ANSWER
    st.write(result["answer"])

    # ✅ SHOW SOURCES ONLY IF NOT SUMMARY
    if query_len <= 8:
        st.markdown("### 📚 Sources")
        for s in result["sources"][:3]:
            st.markdown(f"📄 Page {s['page']}")
            st.info(s["text"][:300])

    # ✅ SAVE HISTORY
    st.session_state.chat_history.insert(0, {
        "question": user_input,
        "answer": result["answer"],
        "source": [
            {"page": s["page"], "text": s["text"][:80]}
            for s in result["sources"][:2]
        ]
    })

    st.session_state.chat_history = st.session_state.chat_history[:10]

if "result" in st.session_state:

    st.markdown("### 🌐 Translate")

    lang_choice = st.selectbox("Language", ["English","Telugu","Hindi"])

    if st.button("Translate"):

        # ✅ Clean answer
        clean = re.sub(r'\[SOURCE:.*?\]', '', st.session_state.result["answer"]).strip()

        # ✅ Limit length (VERY IMPORTANT)
        clean = clean[:2000]

        try:
            if lang_choice == "English":
                st.write(clean)

            elif lang_choice == "Telugu":
                translated = GoogleTranslator(source='auto', target='te').translate(clean)
                st.success(translated)

            elif lang_choice == "Hindi":
                translated = GoogleTranslator(source='auto', target='hi').translate(clean)
                st.success(translated)

        except Exception as e:
            st.error("⚠️ Translation failed. Try again.")