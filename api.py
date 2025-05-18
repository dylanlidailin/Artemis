# api.py
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
# … all your LangChain imports …

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
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# 1) build your llm, parser, loader, retriever, chain
llm    = ChatOpenAI(openai_api_key=API_KEY, model_name="gpt-4")
parser = StrOutputParser()
loader = PyPDFLoader("Knn and Prob-1.pdf")
pages  = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=40).split_documents(loader.load_and_split())
vectorstore = FAISS.from_documents(pages, OpenAIEmbeddings())
retriever   = vectorstore.as_retriever()
prompt      = PromptTemplate.from_template(template=question_retriever)
chain       = RunnableParallel(context=retriever, question=RunnablePassthrough()) | prompt | llm | parser

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(q: Query):
    return {"answer": chain.invoke(q.question)}
