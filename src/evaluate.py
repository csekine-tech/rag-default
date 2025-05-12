import os
from typing import List, Dict
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset
from rag import RAG

# 環境変数の読み込み
load_dotenv()

def create_evaluation_dataset(rag: RAG, test_data: List[Dict]) -> Dataset:
    """評価用データセットを作成"""
    questions = []
    ground_truths = []
    contexts = []
    answers = []

    for item in test_data:
        question = item["question"]
        ground_truth = item["ground_truth"]

        # RAGシステムで回答を生成
        response = rag.query(question)

        questions.append(question)
        ground_truths.append(ground_truth)
        # コンテキストを文字列のリストとして提供
        contexts.append([response["context"]])
        answers.append(response["answer"])

    return Dataset.from_dict({
        "question": questions,
        "ground_truth": ground_truths,
        "contexts": contexts,
        "answer": answers
    })

def main():
    # RAGシステムの初期化
    rag = RAG()

    # テストデータの読み込み
    from data.evaluation.test_dataset import EVALUATION_DATASET

    # 評価用データセットの作成
    eval_dataset = create_evaluation_dataset(rag, EVALUATION_DATASET)

    # 評価指標の定義
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    # 評価の実行
    result = evaluate(
        eval_dataset,
        metrics=metrics,
    )

    # 結果の表示
    print("\n=== RAGAS Evaluation Results ===")
    for metric, score in zip(metrics, result.scores):
        # dict型かつ"score"キーがある場合
        if isinstance(score, dict):
            if "score" in score and isinstance(score["score"], (float, int)):
                print(f"{metric.name}: {score['score']:.4f}")
            else:
                print(f"{metric.name}: {score}")
        # float/intの場合
        elif isinstance(score, (float, int)):
            print(f"{metric.name}: {score:.4f}")
        # それ以外（Noneやstrなど）
        else:
            print(f"{metric.name}: {score}")

if __name__ == "__main__":
    main()