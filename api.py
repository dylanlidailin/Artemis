# api.py

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
from langchain.chains import LLMChain

# To initialize the agent
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.memory import ConversationBufferMemory

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

search = DuckDuckGoSearchAPIWrapper()

# Default placeholder function until PDF is loaded
def run_pdf_qa(q):
    if rag_chain:
        return rag_chain.run(q)
    return "No PDF has been uploaded yet."

tools = [
    Tool(
        name="PDF QA",
        func=run_pdf_qa,
        description="Useful for answering questions about the uploaded PDF document."
    ),
    Tool(
        name="Web Search",
        func=search.run,
        description="Useful for answering general knowledge or current event questions."
    ),
    Tool(
        name="General Chat",
        func=lambda q: llm.invoke(q).content,
        description="Use this for general conversation or questions not related to PDF or search."
    )
]

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

def build_chain_from_pdf(path: str) -> RetrievalQA:
    # 1) Load & split into smaller chunks
    loader = PyPDFLoader(path)
    docs = loader.load_and_split()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,        # smaller chunks
        chunk_overlap=100
    )
    pages = splitter.split_documents(docs)

    # 2) Embed & index
    embeddings = OpenAIEmbeddings()
    index = FAISS.from_documents(pages, embeddings)
    retriever = index.as_retriever()

    # 3) Map and combine prompts
    QUESTION_PROMPT = PromptTemplate(
        template="""
You are given a question and one document chunk. 
Produce a concise answer based *only* on that chunk.
If it doesn’t contain the answer, say “No answer here.”
Question: {question}
=========
Chunk:
{context}
""",
        input_variables=["question", "context"],
    )

    COMBINE_PROMPT = PromptTemplate(
        template="""
You are given the question and multiple intermediate answers.
Combine them into a final, coherent answer.
Question: {question}
Intermediate answers:
{summaries}
""",
        input_variables=["question", "summaries"],
    )

    # 4) Build a map_reduce RetrievalQA
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={
            "question_prompt": QUESTION_PROMPT,
            "combine_prompt": COMBINE_PROMPT,
        },
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
    return {"status": "Uploaded", "filename": file.filename}

# Question endpoint
def is_pdf_question(question: str) -> bool:
    keywords = ["this document", "pdf", "section", "clause", "page", "in the file", "contract"]
    return any(kw in question.lower() for kw in keywords)

# Question endpoint
class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(q: Query):
    try:
        answer = agent_executor.run(q.question)
    except Exception as e:
        answer = f"Agent error: {str(e)}"
    return {"answer": answer}
