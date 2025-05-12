import logging
from typing import List, Optional
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from rag.embeddings.embedding_manager import EmbeddingManager

class RetrieverError(Exception):
    """
    Retriever関連の例外クラス。
    """
    pass

class Retriever:
    """
    ベクトルストアを用いた文書検索・追加を担当するクラス。
    vectorstoreはDI可能。
    """
    def __init__(
        self,
        embedding_manager: Optional[EmbeddingManager] = None,
        vectorstore: Optional[Chroma] = None
    ):
        """
        Args:
            embedding_manager (Optional[EmbeddingManager]): 埋め込み管理器
            vectorstore (Optional[Chroma]): ベクトルストアインスタンス
        """
        if vectorstore is not None:
            self.vectorstore = vectorstore
        elif embedding_manager is not None:
            self.vectorstore = embedding_manager.vector_store
        else:
            raise RetrieverError("vectorstoreまたはembedding_managerのいずれかが必要です。")

    def add_documents(self, documents: List[Document]) -> None:
        """
        文書をベクトルストアに追加し、永続化する。
        Args:
            documents (List[Document]): 追加する文書
        """
        try:
            self.vectorstore.add_documents(documents)
            self.vectorstore.persist()
        except Exception as e:
            logging.error(f"文書追加中にエラー: {e}")
            raise RetrieverError(f"文書追加中にエラー: {e}")

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        similarity_threshold: float = 0.7
    ) -> str:
        """
        クエリに関連する文書を検索し、スコア閾値でフィルタしてコンテキストを返す。
        Args:
            query (str): クエリテキスト
            top_k (int): 上位k件取得
            similarity_threshold (float): 類似度スコアの閾値
        Returns:
            str: フィルタ済み文書のコンテキスト
        """
        try:
            docs = self.vectorstore.similarity_search_with_score(
                query,
                k=top_k
            )
            filtered_docs = [
                doc for doc, score in docs
                if score >= similarity_threshold
            ]
            if not filtered_docs:
                return "関連する情報が見つかりませんでした。"
            context = "\n\n".join([doc.page_content for doc in filtered_docs])
            return context
        except Exception as e:
            logging.error(f"検索中にエラー: {e}")
            raise RetrieverError(f"検索中にエラー: {e}")

# TODO: 設定値や入出力の型安全性向上のため、pydantic/dataclassの活用を検討