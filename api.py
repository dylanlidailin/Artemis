# api.py
from fastapi import FastAPI
from dotenv import load_dotenv
import os
# … your langchain imports …

load_dotenv()
app = FastAPI()

@app.post("/ask")
async def ask(q: dict):
    # your chain.invoke code here
    return {"answer": chain.invoke(q["question"])}