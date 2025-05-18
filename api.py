# api.py
from fastapi import FastAPI
from dotenv import load_dotenv
import os
# … your langchain imports …
# load secrets
from dotenv import load_dotenv
import os

# core LLM + parsing
from langchain_openai.chat_models import ChatOpenAI  
from langchain_core.output_parsers import StrOutputParser

# prompt templating
from langchain.prompts import PromptTemplate

# document loading & splitting
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# embeddings & vector store
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# retrieval + orchestration
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

load_dotenv()
app = FastAPI()

@app.post("/ask")
async def ask(q: dict):
    # your chain.invoke code here
    return {"answer": chain.invoke(q["question"])}