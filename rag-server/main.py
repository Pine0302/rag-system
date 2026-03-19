import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import logging
logging.basicConfig(level=logging.INFO)

from fastapi import FastAPI
from ingest import build_index
from query_engine import get_query_engine, retrieve_with_debug

app = FastAPI()

query_engine = None
reranker = None


@app.on_event("startup")
def startup():
    global query_engine, reranker

    print("Skipping build_index at startup (run manually)")

    print("Starting get_query_engine...")
    query_engine, reranker = get_query_engine()
    print("get_query_engine completed")


@app.get("/")
def health():
    return {"status": "rag server running"}


@app.post("/query")
def query_question(q: str):
    """
    查询接口
    1. 从 Qdrant 召回 20 条
    2. reranker 精排保留 top 3
    3. 打印调试日志
    """
    # 获取精排后的 top 3 节点
    top_nodes = retrieve_with_debug(query_engine, reranker, q)

    # 使用精排后的节点执行查询
    # 通过修改 retriever 的 nodes 来使用预取的节点
    original_retrieve = query_engine._retriever.retrieve
    query_engine._retriever.retrieve = lambda x: top_nodes

    response = query_engine.query(q)

    # 恢复原始 retriever
    query_engine._retriever.retrieve = original_retrieve

    return {
        "question": q,
        "answer": str(response)
    }