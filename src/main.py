from src.rag.embeddings.embedding_manager import EmbeddingManager
from src.rag.generation.answer_generator import AnswerGenerator
from src.config import get_settings
import os

def main():
    settings = get_settings()

    # コンポーネントの初期化
    embedding_manager = EmbeddingManager()
    answer_generator = AnswerGenerator()

    # ナレッジベースディレクトリの確認
    if not os.path.exists(settings.KNOWLEDGE_BASE_DIRECTORY):
        print(f"エラー: ナレッジベースディレクトリ '{settings.KNOWLEDGE_BASE_DIRECTORY}' が存在しません。")
        return

    # ドキュメントのロードと埋め込みの作成
    print("ドキュメントをロード中...")
    texts = embedding_manager.load_documents(settings.KNOWLEDGE_BASE_DIRECTORY)
    if texts:
        print(f"{len(texts)}件のドキュメントを読み込みました。")
        print("埋め込みを作成中...")
        embedding_manager.create_embeddings(texts)
        print("埋め込みの作成が完了しました。")
    else:
        print("警告: 読み込めるドキュメントが見つかりませんでした。")
        return

    # 対話ループ
    print("\n質問を入力してください（終了するには 'quit' と入力）:")
    while True:
        question = input("\n質問: ").strip()
        if question.lower() == 'quit':
            break

        # 類似ドキュメントの検索
        context = embedding_manager.search_similar(question)

        # 回答の生成
        answer = answer_generator.generate_answer(question, context)
        print("\n回答:", answer)

if __name__ == "__main__":
    main()