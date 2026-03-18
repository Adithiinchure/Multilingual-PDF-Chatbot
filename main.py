import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import sys
import shutil

from langdetect import detect
from deep_translator import GoogleTranslator

load_dotenv()

if not os.getenv("OPENROUTER_API_KEY"):
    print("OPENROUTER_API_KEY missing in .env file")
    sys.exit(1)

# -------------------------------
# Step 1: Load PDF
# -------------------------------

print("Loading PDF...")

pdf_path = r"C:\Users\inchu\OneDrive\Desktop\docs'c advice\India_Nature_Culture_History_Trilingual.pdf"
pdf_name = os.path.basename(pdf_path)

documents = []

try:

    with pdfplumber.open(pdf_path) as pdf:

        for page_num, page in enumerate(pdf.pages):

            extracted = page.extract_text()

            if extracted:
                documents.append({
                    "text": extracted,
                    "page": page_num + 1,
                    "source": pdf_name
                })

    if not documents:
        print("No text extracted from PDF.")
        sys.exit(1)

    print("PDF loaded successfully")

except Exception as e:
    print(f"PDF Error: {e}")
    sys.exit(1)

# -------------------------------
# Step 2: Split text
# -------------------------------

# ⭐ REDUCED CHUNK SIZE
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=400
)

texts = []
metadatas = []

for doc in documents:

    chunks = splitter.split_text(doc["text"])

    for chunk in chunks:
        texts.append(chunk)
        metadatas.append({
            "page": doc["page"],
            "source": doc["source"]
        })

print(f"Total Chunks Created: {len(texts)}")

# -------------------------------
# Step 3: Vector Database
# -------------------------------

if os.path.exists("chroma_db"):
    shutil.rmtree("chroma_db")

print("Initializing Vector DB...")

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base"
)

vectordb = Chroma.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadatas
)

# ⭐ REDUCED RETRIEVAL
retriever = vectordb.as_retriever(search_kwargs={"k": 2})

print("Vector DB ready")

# -------------------------------
# Step 4: LLM
# -------------------------------

llm = ChatOpenAI(
    model="Llama 3.1 70B Versatile",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.1,
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "Multilingual-RAG"
    }
)

# -------------------------------
# Step 5: Prompt
# -------------------------------

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant answering questions from a document.

Use ONLY the information from the provided context.

If the answer is not present in the context say:
"Not enough information in the document."

Context:
{context}

Question:
{question}

Answer in English:
"""
)

chain = prompt | llm | StrOutputParser()

print("\nReady! Ask questions.")
print("Type 'exit' to quit.\n")

# -------------------------------
# Step 6: Q&A Loop
# -------------------------------

while True:

    question = input("Your Question: ")

    if question.lower() in ["exit", "quit"]:
        print("Bye!")
        break

    # Detect language
    try:
        lang = detect(question)
    except:
        lang = "en"

    # Translate question to English
    if lang != "en":
        translated_q = GoogleTranslator(source='auto', target='en').translate(question)
        print(f"\nTranslated Question → {translated_q}")
    else:
        translated_q = question

    docs = retriever.invoke(translated_q)

    if not docs:
        print("Answer: Not enough info in document.")
        continue

    # ⭐ SAFETY LIMIT
    docs = docs[:2]

    context = "\n\n".join([d.page_content[:800] for d in docs])

    # ⭐ CONTEXT SIZE CHECK
    if len(context) > 3000:
        context = context[:3000]

    print("Answering...")

    try:
        answer = chain.invoke({
            "context": context,
            "question": translated_q
        })

    except Exception:
        print("Model error, retrying once...")

        try:
            answer = chain.invoke({
                "context": context,
                "question": translated_q
            })

        except Exception:
            print("Model failed again.")
            answer = "Model server error. Please try again."

    print("\nAnswer:", answer)

    # Sources
    if "Not enough info" not in answer:

        sources = set(
            f"{d.metadata['source']} (Page {d.metadata['page']})"
            for d in docs
        )

        print("\nSources:")
        for s in sources:
            print("-", s)

    else:
        print("\nNo valid source found in document.")

    # Translation options
    print("\nTranslation Options:")
    print("1 → Telugu")
    print("2 → English")
    print("3 → Hindi")
    print("4 → Skip")

    choice = input("Choose option: ")

    if choice == "1":
        translated = GoogleTranslator(source='auto', target='te').translate(answer)
        print("\nAnswer (Telugu):", translated)

    elif choice == "2":
        translated = GoogleTranslator(source='auto', target='en').translate(answer)
        print("\nAnswer (English):", translated)

    elif choice == "3":
        translated = GoogleTranslator(source='auto', target='hi').translate(answer)
        print("\nAnswer (Hindi):", translated)

    else:
        print("Skipping translation.")