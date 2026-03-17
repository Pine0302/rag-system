from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

VAULT_PATH = "/vault/obsidian"

def build_index():

    documents = SimpleDirectoryReader(
        VAULT_PATH,
        recursive=True,
        required_exts=[".md", ".qmd"],
        exclude=[".obsidian"]
    ).load_data()

    embed_model = HuggingFaceEmbedding(
        model_name="/models/BAAI/bge-base-zh-v1.5",
        cache_folder="/models"
    )

    client = QdrantClient(
        host="qdrant",
        port=6333
    )

    # 删除并重新创建集合
    try:
        client.delete_collection(collection_name="obsidian_notes")
    except:
        pass

    client.create_collection(
        collection_name="obsidian_notes",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name="obsidian_notes"
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
        storage_context=storage_context
    )

    return index
