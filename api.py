import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pypdf
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import uuid


@dataclass
class PdfSession:
    llm: ChatGoogleGenerativeAI
    retriever: Any


SESSION_STORE: dict[str, PdfSession] = {}
load_dotenv()


def get_gemini_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set GEMINI_API_KEY in your .env")
    return api_key


def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(google_api_key=get_gemini_api_key(), model="gemini-3.5-flash")

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_file = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_file.read_text())

# === PDF QA Tool ===
def build_chain_from_pdf(path: str) -> PdfSession:
    from langchain_community.vectorstores import FAISS
    with open(path, "rb") as pdf_file:
        reader = pypdf.PdfReader(pdf_file)
        docs_text = ""
        for page in reader.pages:
            docs_text += page.extract_text()
    
    from langchain_core.documents import Document
    docs = [Document(page_content=docs_text)]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    pages = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    index = FAISS.from_documents(pages, embeddings)
    retriever = index.as_retriever()
    llm = get_llm()

    return PdfSession(llm=llm, retriever=retriever)

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    path = None
    try:
        suffix = Path(file.filename).suffix or ".pdf"
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp.flush()
            path = tmp.name

        session = build_chain_from_pdf(path)
        if session is None:
            raise HTTPException(status_code=400, detail="无法构建 chain")

        session_id = str(uuid.uuid4())
        SESSION_STORE[session_id] = session

        return {
            "status": "Uploaded",
            "filename": file.filename,
            "session_id": session_id
        }

    except Exception as e:
        print("[PDF analysis error]", str(e))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass

# === Ask endpoint ===
class Query(BaseModel):
    question: str
    session_id: str

@app.post("/ask")
async def ask(q: Query):
    session = SESSION_STORE.get(q.session_id)

    if session is None:
        return {"answer": "Session not found or expired."}

    try:
        docs = session.retriever.invoke(q.question)
        context = "\n".join([doc.page_content for doc in docs])
        
        prompt = PromptTemplate(
            template="""Answer the question based on this context:
{context}

Question: {question}""",
            input_variables=["context", "question"]
        )
        
        chain = prompt | session.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": q.question})
    except Exception as e:
        answer = f"Agent error: {str(e)}"

    return {"answer": answer}

@app.post("/extract_keywords")
async def extract_keywords(q: Query):
    session = SESSION_STORE.get(q.session_id)
    if session is None:
        return {"answer": "Session not found or expired."}

    try:
        docs = session.retriever.invoke(q.question)
        context = " ".join([doc.page_content for doc in docs])

        template = """You are an expert in resume analysis.
Analyze the provided resume and extract key skills, technologies, and experience.
Format the output as a comma-separated list of keywords.

Resume content:
{context}"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context"],
        )

        chain = prompt | session.llm | StrOutputParser()
        answer = chain.invoke({"context": context})

    except Exception as e:
        answer = f"Agent error: {str(e)}"

    return {"answer": answer}

@app.get("/healthz")
async def health():
    return {"status": "ok"}