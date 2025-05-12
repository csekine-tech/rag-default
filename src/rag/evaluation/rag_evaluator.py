from typing import List, Dict, Any
from langchain.schema import Document
from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ..embeddings.embedding_manager import EmbeddingManager
from ..generation.answer_generator import AnswerGenerator
from config import get_settings, get_generation_model_config

class RAGEvaluator:
    def __init__(self, embedding_manager: EmbeddingManager, answer_generator: AnswerGenerator):
        self.embedding_manager = embedding_manager
        self.answer_generator = answer_generator

        # 評価用のLLMを作成
        settings = get_settings()
        model_config = get_generation_model_config()
        self.eval_llm = ChatOpenAI(
            model=model_config.model_name,
            api_key=model_config.openai_api_key,
            temperature=0.0
        )

        # メトリクスの設定
        self.metrics = [
            Faithfulness(llm=self.eval_llm),
            AnswerRelevancy(llm=self.eval_llm),
            ContextPrecision(llm=self.eval_llm),
            ContextRecall(llm=self.eval_llm)
        ]

    def evaluate_response(self, question: str, ground_truth: str) -> Dict[str, float]:
        # 類似文書の検索
        similar_docs = self.embedding_manager.search_similar(question)

        # 回答の生成
        answer = self.answer_generator.generate_answer(question, similar_docs)

        # 評価用データセットの作成
        data = {
            "question": [question],
            "answer": [answer],
            "ground_truth": [ground_truth],
            "contexts": [[doc.page_content for doc in similar_docs]]
        }
        dataset = Dataset.from_dict(data)

        # 評価の実行
        results = evaluate(
            dataset=dataset,
            metrics=self.metrics
        )

        return results.to_dict()

    def evaluate_dataset(self, dataset: List[Dict[str, str]]) -> Dict[str, float]:
        questions = []
        ground_truths = []
        answers = []
        contexts = []

        for item in dataset:
            question = item["question"]
            ground_truth = item["ground_truth"]

            # 類似文書の検索
            similar_docs = self.embedding_manager.search_similar(question)

            # 回答の生成
            answer = self.answer_generator.generate_answer(question, similar_docs)

            questions.append(question)
            ground_truths.append(ground_truth)
            answers.append(answer)
            contexts.append([doc.page_content for doc in similar_docs])

        # 評価用データセットの作成
        data = {
            "question": questions,
            "answer": answers,
            "ground_truth": ground_truths,
            "contexts": contexts
        }
        dataset = Dataset.from_dict(data)

        # 評価の実行
        results = evaluate(
            dataset=dataset,
            metrics=self.metrics
        )

        return results.to_dict()