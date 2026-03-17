import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import logging
logging.basicConfig(level=logging.INFO)

from fastapi import FastAPI
from ingest import build_index
from query_engine import get_query_engine

app = FastAPI()

index = None
query_engine = None


@app.on_event("startup")
def startup():

    global index
    global query_engine

    print("Starting build_index...")
    index = build_index()
    print("build_index completed")

    print("Starting get_query_engine...")
    query_engine = get_query_engine()
    print("get_query_engine completed")


@app.get("/")
def health():
    return {"status": "rag server running"}


@app.post("/query")
def query_question(q: str):

    response = query_engine.query(q)

    return {
        "question": q,
        "answer": str(response)
    }
