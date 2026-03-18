from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient

# 禁用 LLM
Settings.llm = None

def get_query_engine():

    embed_model = HuggingFaceEmbedding(
        model_name="/models/BAAI/bge-m3",
        cache_folder="/models"
    )

    client = QdrantClient(
        host="qdrant",
        port=6333
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name="obsidian_notes"
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model
    )

    # 明确设置 llm=None 来禁用 LLM
    return index.as_query_engine(llm=None)
