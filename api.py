import os
from dotenv import load_dotenv

# FastAPI imports
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from tempfile import NamedTemporaryFile
from fastapi.staticfiles import StaticFiles

# LangChain imports
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# 1) Load environment and API key
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Please set the OPENAI_API_KEY environment variable in .env")

# 2) Initialize LLM and parser
llm = ChatOpenAI(openai_api_key=API_KEY, model_name="gpt-4")
parser = StrOutputParser()

# 3) Define prompt template for RAG
question_retriever = '''
You are a helpful assistant. You will be given a question and a context.
Your task is to answer the question based on the context provided.
If the context does not contain enough information to answer the question, say "I don't know".

context: {context}
question: {question}
'''

# 4) Create FastAPI app and serve static frontend
app = FastAPI()
app.mount("/static", StaticFiles(directory="static", html=True), name="static_files")

# 5) Globals for chain and vectorstore
chain = None
vectorstore = None

# 6) Helper to build or rebuild RAG chain
def build_chain(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load_and_split()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    pages = splitter.split_documents(docs)
    vs = FAISS.from_documents(pages, OpenAIEmbeddings())
    retriever = vs.as_retriever()
    prompt = PromptTemplate.from_template(question_retriever)
    rag_chain = (
        RunnableParallel(context=retriever, question=RunnablePassthrough())
        | prompt
        | llm
        | parser
    )
    return rag_chain, vs

# 7) Endpoint: upload PDF to reindex
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global chain, vectorstore
    with NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        chain, vectorstore = build_chain(tmp.name)
    return {"status": "uploaded", "filename": file.filename}

# Pydantic model for queries
class Query(BaseModel):
    question: str

# 8) Endpoint: ask a question against current chain
@app.post("/ask")
async def ask(q: Query):
    if chain is None:
        return {"error": "No PDF loaded. Please upload one via /upload_pdf."}
    answer = chain.invoke(q.question)
    return {"answer": answer}

# 9) Simple GET root message if someone visits
@app.get("/api-info")
async def api_info():
    return {"message": "PDF Chatbot API: POST /upload_pdf then POST /ask"}