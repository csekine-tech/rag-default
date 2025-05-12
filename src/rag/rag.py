import logging
from typing import Dict, Optional
from rag.embeddings.embedding_manager import EmbeddingManager
from rag.generation.answer_generator import AnswerGenerator
from rag.retrieval.retriever import Retriever

class RAGError(Exception):
    """
    RAG統合層の例外クラス。
    """
    pass

class RAG:
    """
    EmbeddingManager, AnswerGenerator, Retrieverを統合し、RAG全体のフローを管理するクラス。
    各コンポーネントはDI可能。
    """
    def __init__(
        self,
        embedding_manager: Optional[EmbeddingManager] = None,
        answer_generator: Optional[AnswerGenerator] = None,
        retriever: Optional[Retriever] = None
    ):
        """
        Args:
            embedding_manager (Optional[EmbeddingManager]): 埋め込み管理器
            answer_generator (Optional[AnswerGenerator]): 回答生成器
            retriever (Optional[Retriever]): 検索器
        """
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.answer_generator = answer_generator or AnswerGenerator()
        self.retriever = retriever or Retriever(self.embedding_manager)

    def query(self, question: str) -> Dict[str, str]:
        """
        質問に対する回答を生成する。
        Args:
            question (str): 質問文
        Returns:
            Dict[str, str]: answer, contextを含む辞書
        Raises:
            RAGError: 検索・生成時の例外
        """
        try:
            # 関連文書の検索
            context = self.retriever.retrieve(question)
            # 回答の生成
            answer = self.answer_generator.generate(question, context)
            return {
                "answer": answer,
                "context": context
            }
        except Exception as e:
            logging.error(f"RAG統合処理中にエラー: {e}")
            raise RAGError(f"RAG統合処理中にエラー: {e}")

# TODO: 戻り値の型安全性向上のため、pydantic/dataclassの活用を検討