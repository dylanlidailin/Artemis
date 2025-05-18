# api.py
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

# LangChain imports
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# 1) Load your .env and get the key
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Please set the OPENAI_API_KEY environment variable in .env")

# 2) Build your RAG chain
llm = ChatOpenAI(openai_api_key=API_KEY, model_name="gpt-4")
parser = StrOutputParser()

loader = PyPDFLoader("Knn and Prob-1.pdf")
docs = loader.load_and_split()
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=40)
pages = splitter.split_documents(docs)

vectorstore = FAISS.from_documents(pages, OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

question_retriever = """
You are a helpful assistant. You will be given a question and a context.
Your task is to answer the question based on the context provided.
If the context does not contain enough information to answer the question, say "I don't know".

context: {context}
question: {question}
"""

prompt = PromptTemplate.from_template(question_retriever)
chain = (
    RunnableParallel(context=retriever, question=RunnablePassthrough())
    | prompt
    | llm
    | parser
)

# 3) Expose it via FastAPI
app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(q: Query):
    answer = chain.invoke(q.question)
    return {"answer": answer}

@app.get("/")
async def read_root():
    return {"message": "Welcome to the PDF-chatbot API. Use POST /ask to query."}
