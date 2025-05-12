from typing import List, Dict, Optional
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from src.rag.embeddings.embedding_manager import EmbeddingManager

class Retriever:
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.vectorstore = embedding_manager.vector_store

    def add_documents(self, documents: List[Document]) -> None:
        """文書をベクトルストアに追加"""
        self.vectorstore.add_documents(documents)
        self.vectorstore.persist()

    def retrieve(self, query: str, top_k: int = 3, similarity_threshold: float = 0.7) -> str:
        """クエリに関連する文書を検索"""
        # 類似度に基づいて文書を検索
        docs = self.vectorstore.similarity_search_with_score(
            query,
            k=top_k
        )

        # 類似度スコアでフィルタリング
        filtered_docs = [
            doc for doc, score in docs
            if score >= similarity_threshold
        ]

        if not filtered_docs:
            return "関連する情報が見つかりませんでした。"

        # コンテキストの生成
        context = "\n\n".join([doc.page_content for doc in filtered_docs])
        return context