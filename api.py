# Import libraries
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load API key
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY in your .env")

llm = ChatOpenAI(openai_api_key=API_KEY, model="gpt-4")

# FastAPI setup
app = FastAPI()

# Allow front-end to call these endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount your static folder (index.html, app.js, css, etc.)
app.mount("/static", StaticFiles(directory="static", html=True), name="static")


# Serve your original index.html at `/`
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_file = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_file.read_text())


# Globals to hold the last‐uploaded PDF and its chain
current_pdf_path: str | None = None
rag_chain: RetrievalQA | None = None


def build_chain_from_pdf(path: str) -> RetrievalQA:
    loader = PyPDFLoader(path)
    docs = loader.load_and_split()
    pages = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    embeddings = OpenAIEmbeddings()
    index = FAISS.from_documents(pages, embeddings)
    retriever = index.as_retriever()
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
    )


# Upload endpoint to rebuild the chain on demand
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global current_pdf_path, rag_chain

    # write to a temp file so PyPDFLoader can read it
    suffix = Path(file.filename).suffix or ".pdf"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        current_pdf_path = tmp.name

    # rebuild the chain
    rag_chain = build_chain_from_pdf(current_pdf_path)
    return {"status": "uploaded", "filename": file.filename}


# Question endpoint
class Query(BaseModel):
    question: str


@app.post("/ask")
async def ask(q: Query):
    global rag_chain
    if rag_chain is None:
        return {"error": "No PDF loaded. Please POST /upload_pdf first."}

    # run  RetrievalQA
    answer = rag_chain.run(q.question)
    return {"answer": answer}