import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import NodeWithScore
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

# 配置 ModelScope 离线模式
os.environ["MODELSCOPE_SDK_NETCARD"] = "offline"
os.environ["MODELSCOPE_ENABLE_CACHE"] = "true"


# 禁用 LLM
Settings.llm = None


class BGEReranker:
    """
    基于 BGE-reranker-base 的重排序器
    使用 sentence_transformers 的 CrossEncoder 实现
    模型从本地缓存加载（ModelScope 下载的模型）
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", cache_folder: str = "/models", top_n: int = 3):
        self.model_name = model_name
        self.cache_folder = cache_folder
        self.top_n = top_n
        self._model = None

    def _get_local_model_path(self) -> str:
        """获取本地模型路径"""
        # ModelScope 下载的模型通常缓存在 /models 下
        # 路径格式: /models/BAAI/bge-reranker-base
        local_path = os.path.join(self.cache_folder, self.model_name)
        if os.path.exists(local_path):
            return local_path
        # 尝试其他可能的路径格式
        alt_path = os.path.join(self.cache_folder, "BAAI", "bge-reranker-base")
        if os.path.exists(alt_path):
            return alt_path
        return self.model_name  # fallback to model name

    @property
    def model(self):
        if self._model is None:
            local_path = self._get_local_model_path()
            logger.info(f"Loading reranker model from local: {local_path}")

            # 直接使用模型路径创建 CrossEncoder
            self._model = CrossEncoder(
                local_path if os.path.exists(local_path) else self.model_name,
                max_length=512
            )
            logger.info("Reranker model loaded successfully from local cache")
        return self._model

    def postprocess_nodes(self, nodes: list, query_str: str) -> list:
        """
        对节点进行重排序
        1. 提取 query 和 node text 成对
        2. 使用 CrossEncoder 计算相关性分数
        3. 按分数排序并返回 top_n
        """
        if not nodes:
            return []

        # 构造 query-document 对
        pairs = [[query_str, node.get_content()] for node in nodes]

        # 计算相关性分数
        scores = self.model.predict(pairs)

        # 将分数附加到节点上
        scored_nodes = []
        for node, score in zip(nodes, scores):
            scored_node = NodeWithScore(
                node=node.node if hasattr(node, 'node') else node,
                score=float(score)
            )
            scored_nodes.append(scored_node)

        # 按分数降序排序
        scored_nodes.sort(key=lambda x: x.score, reverse=True)

        return scored_nodes[:self.top_n]


def get_embed_model():
    """获取 embedding 模型"""
    return HuggingFaceEmbedding(
        model_name="/models/BAAI/bge-m3",
        cache_folder="/models"
    )


def get_reranker():
    """获取 reranker 模型（从本地 ModelScope 缓存加载）"""
    return BGEReranker(
        model_name="BAAI/bge-reranker-base",
        cache_folder="/models",
        top_n=3
    )


def get_query_engine():
    """
    获取 query engine
    - similarity_top_k=20: 从 Qdrant 召回 20 条
    - reranker: 精排后保留 top 3
    """
    embed_model = get_embed_model()

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

    reranker = get_reranker()

    # similarity_top_k=20 召回，reranker 精排
    query_engine = index.as_query_engine(
        llm=None,
        similarity_top_k=20
    )

    return query_engine, reranker


def retrieve_with_debug(query_engine, reranker, query_str: str):
    """
    检索函数，包含调试日志
    1. 从 Qdrant 召回 20 条
    2. 打印 rerank 前 Top5
    3. reranker 精排后打印 Top3
    4. 返回精排后的节点
    """
    # 获取原始召回结果（similarity_top_k=20）
    nodes_with_scores = query_engine._retriever.retrieve(query_str)

    # 打印 rerank 前 Top5
    logger.info("=" * 50)
    logger.info("【Rerank Debug】Rerank 前 Top5 (原始召回):")
    for i, node in enumerate(nodes_with_scores[:5]):
        file_path = node.metadata.get("source", "unknown") if node.metadata else "unknown"
        score = node.score if hasattr(node, 'score') else "N/A"
        logger.info(f"  [{i+1}] file_path={file_path}, score={score:.4f}" if isinstance(score, float) else f"  [{i+1}] file_path={file_path}, score={score}")

    # 执行 rerank（top_n=3）
    reranked_nodes = reranker.postprocess_nodes(nodes_with_scores, query_str)

    # 打印 rerank 后 Top3
    logger.info("【Rerank Debug】Rerank 后 Top3 (精排结果):")
    for i, node in enumerate(reranked_nodes[:3]):
        file_path = node.metadata.get("source", "unknown") if node.metadata else "unknown"
        score = node.score if hasattr(node, 'score') else "N/A"
        logger.info(f"  [{i+1}] file_path={file_path}, score={score:.4f}" if isinstance(score, float) else f"  [{i+1}] file_path={file_path}, score={score}")
    logger.info("=" * 50)

    return reranked_nodes[:3]