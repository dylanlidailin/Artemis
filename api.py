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
from langchain.prompts import PromptTemplate

from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.utilities import SerpAPIWrapper

import uuid

SESSION_STORE = {}
load_dotenv()
SERP_API_KEY = os.getenv("SERPAPI_API_KEY")

search = SerpAPIWrapper(serpapi_api_key=SERP_API_KEY)

search_tool = Tool(
    name="SerpAPI Search",
    func=search.run,
    description="Useful for answering questions about current events or real-time web data"
)

# Load API key
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY in your .env")

llm = ChatOpenAI(openai_api_key=API_KEY, model="gpt-4")

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
def build_chain_from_pdf(path: str) -> RetrievalQA:
    loader = PyPDFLoader(path)
    docs = loader.load_and_split()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    pages = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    index = FAISS.from_documents(pages, embeddings)
    retriever = index.as_retriever()

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

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={
            "question_prompt": question_prompt,
            "combine_prompt": combine_prompt,
        },
    )

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix or ".pdf"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        path = tmp.name

    chain = build_chain_from_pdf(path)

    if chain is None:
        return {"status": "Failed to process PDF."}

    # Generate a session ID
    session_id = str(uuid.uuid4())
    SESSION_STORE[session_id] = chain

    return {
        "status": "Uploaded",
        "filename": file.filename,
        "session_id": session_id
    }

# === Ask endpoint ===
class Query(BaseModel):
    question: str
    session_id: str

@app.post("/ask")
async def ask(q: Query):
    chain = SESSION_STORE.get(q.session_id)

    if chain is None:
        return {"answer": "Session not found or expired."}

    try:
        answer = chain.run(q.question)
    except Exception as e:
        answer = f"Agent error: {str(e)}"

    return {"answer": answer}

if __name__ == "__main__":
    # Render 通常会提供 PORT 环境变量，如果没有则默认使用 8000 端口
    # 这对于在本地开发和在 Render 上部署都很重要
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)