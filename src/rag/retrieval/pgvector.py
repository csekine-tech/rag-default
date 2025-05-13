import logging
from typing import List, Optional
from langchain_postgres import PGVector
from langchain_core.documents import Document

class PgvectorRetrieverError(Exception):
    """
    PgvectorRetriever関連の例外クラス。
    """
    pass

class PgvectorRetriever:
    """
    pgvector(PostgreSQL)を用いた文書検索・追加を担当するRetrieverクラス。
    VectorStoreはDI可能。
    """
    def __init__(
        self,
        vector_store: PGVector,
    ):
        """
        Args:
            vector_store (PGVector): ベクトルストアインスタンス
        """
        self.vector_store = vector_store
        self.logger = logging.getLogger(__name__)

    def add_documents(self, documents: List[Document]) -> None:
        """
        文書をベクトルストアに追加。
        Args:
            documents (List[Document]): 追加する文書
        """
        try:
            self.vector_store.add_documents(documents)
        except Exception as e:
            self.logger.error(f"文書追加中にエラー: {e}")
            raise PgvectorRetrieverError(f"文書追加中にエラー: {e}")

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        similarity_threshold: float = 0.7
    ) -> List[Document]:
        """
        クエリに関連する文書を検索し、スコア閾値でフィルタして返す。
        Args:
            query (str): クエリテキスト
            top_k (int): 上位k件取得
            similarity_threshold (float): 類似度スコアの閾値
        Returns:
            List[Document]: フィルタ済み文書リスト
        """
        try:
            docs_with_score = self.vector_store.similarity_search_with_score(query, k=top_k)
            filtered_docs = [doc for doc, score in docs_with_score if score >= similarity_threshold]
            return filtered_docs
        except Exception as e:
            self.logger.error(f"検索中にエラー: {e}")
            raise PgvectorRetrieverError(f"検索中にエラー: {e}")