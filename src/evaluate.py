from src.rag.embeddings.embedding_manager import EmbeddingManager
from src.rag.generation.answer_generator import AnswerGenerator
from src.rag.evaluation.rag_evaluator import RAGEvaluator
from data.evaluation.test_dataset import EVALUATION_DATASET
import json

def main():
    # コンポーネントの初期化
    embedding_manager = EmbeddingManager()
    answer_generator = AnswerGenerator()
    evaluator = RAGEvaluator(embedding_manager, answer_generator)

    print("RAGシステムの評価を開始します...")

    # データセット全体の評価
    print("\nデータセット全体の評価:")
    dataset_results = evaluator.evaluate_dataset(EVALUATION_DATASET)
    print(json.dumps(dataset_results, indent=2, ensure_ascii=False))

    # 個別の質問に対する評価
    print("\n個別の質問に対する評価:")
    for item in EVALUATION_DATASET:
        print(f"\n質問: {item['question']}")
        result = evaluator.evaluate_response(item["question"], item["ground_truth"])
        print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()