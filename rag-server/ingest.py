from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from chunker import chunk_directory

VAULT_PATH = "/vault/obsidian"

def build_index():

    # Use chunker instead of SimpleDirectoryReader
    documents = chunk_directory(VAULT_PATH)
    print(f"Total chunks: {len(documents)}")

    if not documents:
        print("No documents to index!")
        return None

    embed_model = HuggingFaceEmbedding(
        model_name="/models/BAAI/bge-m3",
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
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )

    # Create payload indexes for metadata fields to enable filtering
    client.create_payload_index(
        collection_name="obsidian_notes",
        field_name="metadata.source",
        field_schema="keyword"
    )
    client.create_payload_index(
        collection_name="obsidian_notes",
        field_name="metadata.title",
        field_schema="keyword"
    )
    client.create_payload_index(
        collection_name="obsidian_notes",
        field_name="metadata.heading_level",
        field_schema="integer"
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name="obsidian_notes"
    )

    # Create index from vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model
    )

    # Insert documents one by one
    for i, doc in enumerate(documents):
        index.insert(doc)
        if (i + 1) % 50 == 0:
            print(f"Indexed {i + 1}/{len(documents)} chunks")

    print(f"Indexed {len(documents)} chunks total")

    # Verify
    info = client.get_collection("obsidian_notes")
    print(f"Points in Qdrant: {info.points_count}")

    return index