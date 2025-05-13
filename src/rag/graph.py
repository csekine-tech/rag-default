"""
LangGraphベースRAGワークフロー雛形

- Retrieverノード
- Rerankノード
- LLMノード
- Postprocessノード
- 評価ノード

今後、各ノードの詳細実装・依存注入・LangSmith連携を追加予定。
"""
from typing import Any, Dict, TypedDict, Optional, List
from langgraph.graph import StateGraph
from langgraph.graph import START
import logging
from .retrieval.pgvector import PgvectorRetriever
from langchain_postgres import PGVector
from src.config import get_settings
from .embeddings.embedding_manager import EmbeddingManager
from .generation.answer_generator import AnswerGenerator

logger = logging.getLogger(__name__)

class RagState(TypedDict, total=False):
    query: str
    retrieved_docs: Optional[List[Any]]
    reranked_docs: Optional[List[Any]]
    generated_answer: Optional[str]
    metrics: Optional[dict]


def retriever_node(state: RagState, retriever: PgvectorRetriever) -> RagState:
    """
    ベクトル検索ノード（pgvector実装）
    """
    logger.info("Retrieverノード実行")
    docs = retriever.retrieve(state["query"])
    state["retrieved_docs"] = docs
    return state


def rerank_node(state: RagState) -> RagState:
    """
    Rerankノード（ダミー実装）
    """
    logger.info("Rerankノード実行")
    state["reranked_docs"] = state.get("retrieved_docs")  # 仮
    return state


def llm_node(state: RagState, answer_generator: AnswerGenerator) -> RagState:
    """
    LLMノード（本実装: AnswerGeneratorを利用）
    """
    logger.info("LLMノード実行")
    docs = state.get("reranked_docs") or state.get("retrieved_docs") or []
    context = "\n\n".join([doc.page_content for doc in docs])
    question = state["query"]
    state["generated_answer"] = answer_generator.generate(question, context)
    return state


def postprocess_node(state: RagState) -> RagState:
    """
    Postprocessノード（ダミー実装）
    """
    logger.info("Postprocessノード実行")
    return state


def metrics_node(state: RagState) -> RagState:
    """
    評価ノード（ダミー実装）
    """
    logger.info("Metricsノード実行")
    state["metrics"] = {"dummy_metric": 1.0}
    return state


def build_rag_graph() -> StateGraph:
    """
    RAGワークフローのLangGraphグラフを構築する。
    Returns:
        StateGraph: LangGraphのグラフインスタンス
    """
    settings = get_settings()
    embedding_manager = EmbeddingManager()
    embedding_function = embedding_manager.embedding_model
    vector_store = PGVector(
        embeddings=embedding_function,
        collection_name=settings.PGVECTOR_COLLECTION,
        connection=f"postgresql+psycopg://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}",
        use_jsonb=True,
    )
    retriever = PgvectorRetriever(vector_store=vector_store)
    answer_generator = AnswerGenerator()

    graph = StateGraph(RagState)
    graph.add_node("retriever", lambda state: retriever_node(state, retriever))
    graph.add_node("rerank", rerank_node)
    graph.add_node("llm", lambda state: llm_node(state, answer_generator))
    graph.add_node("postprocess", postprocess_node)
    graph.add_node("metrics_node", metrics_node)
    # エッジ定義
    graph.add_edge(START, "retriever")
    graph.add_edge("retriever", "rerank")
    graph.add_edge("rerank", "llm")
    graph.add_edge("llm", "postprocess")
    graph.add_edge("postprocess", "metrics_node")
    return graph.compile()