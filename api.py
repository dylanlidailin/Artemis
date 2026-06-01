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
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import uuid


@dataclass
class PdfSession:
    qa_chain: RetrievalQA
    retriever: Any


SESSION_STORE: dict[str, PdfSession] = {}
load_dotenv()


def get_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set OPENAI_API_KEY in your .env")
    return api_key


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(openai_api_key=get_openai_api_key(), model="gpt-4")

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
    loader = PyPDFLoader(path)
    docs = loader.load_and_split()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    pages = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(openai_api_key=get_openai_api_key())
    index = FAISS.from_documents(pages, embeddings)
    retriever = index.as_retriever()
    llm = get_llm()

    question_prompt = PromptTemplate(
        template="""
You are given a question and one document chunk. 
Answer concisely based *only* on that chunk.
If the chunk is irrelevant, respond: "No answer here."
Question: {question}
=========
Chunk:
{context}
""",
        input_variables=["question", "context"]
    )

    combine_prompt = PromptTemplate(
        template="""
You are given a question and multiple intermediate answers.
Combine them into a final, coherent answer.
Question: {question}
Intermediate answers:
{summaries}
""",
        input_variables=["question", "summaries"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={
            "question_prompt": question_prompt,
            "combine_prompt": combine_prompt,
        },
    )

    return PdfSession(qa_chain=qa_chain, retriever=retriever)

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
        answer = session.qa_chain.run(q.question)
    except Exception as e:
        answer = f"Agent error: {str(e)}"

    return {"answer": answer}

@app.post("/extract_keywords")
async def extract_keywords(q: Query):
    session = SESSION_STORE.get(q.session_id)
    if session is None:
        return {"answer": "Session not found or expired."}

    template = """
    You are an expert in resume analysis.
    Analyze the provided resume and extract key skills, technologies, and experience.
    Format the output as a comma-separated list of keywords.

    Resume content:
    {context}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["context"],
    )

    try:
        docs = session.retriever.invoke(q.question)
        context = " ".join([doc.page_content for doc in docs])

        response = get_llm().invoke(prompt.format(context=context))
        answer = getattr(response, "content", str(response))

    except Exception as e:
        answer = f"Agent error: {str(e)}"

    return {"answer": answer}

@app.get("/healthz")
async def health():
    return {"status": "ok"}