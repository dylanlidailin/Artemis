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

from langchain.agents import Tool

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

# Globals
current_pdf_path: str | None = None
rag_chain: RetrievalQA | None = None

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
    global current_pdf_path, rag_chain

    suffix = Path(file.filename).suffix or ".pdf"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        current_pdf_path = tmp.name

    rag_chain = build_chain_from_pdf(current_pdf_path)
    return {"status": "Uploaded", "filename": file.filename}

# === Agent Setup ===
def run_pdf_qa(q: str) -> str:
    if rag_chain is None:
        return "No PDF has been uploaded yet."
    return rag_chain.run(q)

tools = [
    Tool(
        name="PDF QA",
        func=run_pdf_qa,
        description="Answer questions about the uploaded PDF."
    ),
    Tool(
        name="Web Search",
        func=search.run,
        description="Look up current or general info from the web."
    ),
    Tool(
        name="General Chat",
        func=lambda q: llm.invoke(q).content,
        description="Answer general knowledge or personal questions."
    )
]

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent_executor = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# === Ask endpoint ===
class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(q: Query):
    try:
        answer = agent_executor.run(q.question)
    except Exception as e:
        answer = f"Agent error: {str(e)}"
    return {"answer": answer}
