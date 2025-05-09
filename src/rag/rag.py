from typing import Dict, List
from src.rag.embeddings.embedding_manager import EmbeddingManager
from src.rag.generation.answer_generator import AnswerGenerator
from src.rag.retrieval.retriever import Retriever

class RAG:
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.answer_generator = AnswerGenerator()
        self.retriever = Retriever(self.embedding_manager)

    def query(self, question: str) -> Dict[str, str]:
        """質問に対する回答を生成"""
        # 関連文書の検索
        context = self.retriever.retrieve(question)

        # 回答の生成
        answer = self.answer_generator.generate(question, context)

        return {
            "answer": answer,
            "context": context
        }