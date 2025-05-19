# api.py
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("set OPENAI_API_KEY in your .env")

llm = ChatOpenAI(openai_api_key=API_KEY, model_name="gpt-4")

#––– GLOBALS –––
app = FastAPI()

# serve your index.html + static/js out of ./static
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# allow your front‐end (if you host it separately) to talk to this API:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# keep track of the last‐uploaded pdf on disk
current_pdf_path: str | None = None

# and our RAG chain
rag_chain: RetrievalQA | None = None


def build_new_chain(pdf_path: str) -> RetrievalQA:
    """Load a PDF, split it, build a FAISS retriever + RetrievalQA chain."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load_and_split()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    pages = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    index = FAISS.from_documents(pages, embeddings)
    retriever = index.as_retriever()

    # use stuff so that if nothing is found, it still answers from LLM
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
    )


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html.read_text())


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global current_pdf_path, rag_chain

    # save into a temp file
    suffix = Path(file.filename).suffix or ".pdf"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        current_pdf_path = tmp.name

    # build a brand new chain from that file
    rag_chain = build_new_chain(current_pdf_path)

    return {"status": "uploaded", "filename": file.filename}


class QuestionIn(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(q: QuestionIn):
    global rag_chain, current_pdf_path

    # if they hit /ask without ever uploading, error
    if rag_chain is None:
        return {"error": "No PDF is loaded.  Please POST /upload_pdf first."}

    # run your RetrievalQA
    answer = rag_chain.run(q.question)
    return {"answer": answer}