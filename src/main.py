import logging
from rag.embeddings.embedding_manager import EmbeddingManager
from rag.generation.answer_generator import AnswerGenerator
from config import get_settings
from typing import Optional
import os

def setup_logging() -> None:
    """
    ロギング設定を初期化する。
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

def input_loop(answer_generator: AnswerGenerator, embedding_manager: EmbeddingManager) -> None:
    """
    対話型の質問応答ループを実行する。
    Args:
        answer_generator (AnswerGenerator): 回答生成器
        embedding_manager (EmbeddingManager): 埋め込み管理器
    """
    print("\n質問を入力してください（終了するには 'quit' と入力）:")
    while True:
        question = input("\n質問: ").strip()
        if question.lower() == 'quit':
            break
        context = embedding_manager.search_similar(question)
        # contextはList[Document]なので、テキスト結合
        context_text = "\n".join([doc.page_content for doc in context])
        answer = answer_generator.generate(question, context_text)
        print("\n回答:", answer)

def main() -> None:
    """
    RAGシステムのエントリーポイント。
    設定・初期化・入出力・例外ハンドリングのみを担当。
    """
    setup_logging()
    settings = get_settings()
    try:
        embedding_manager = EmbeddingManager()
        answer_generator = AnswerGenerator()
        # ナレッジベースディレクトリの確認
        if not os.path.exists(settings.KNOWLEDGE_BASE_DIRECTORY):
            logging.error(f"ナレッジベースディレクトリ '{settings.KNOWLEDGE_BASE_DIRECTORY}' が存在しません。")
            return
        # ドキュメントのロードと埋め込みの作成
        logging.info("ドキュメントをロード中...")
        texts = embedding_manager.load_documents(settings.KNOWLEDGE_BASE_DIRECTORY)
        if texts:
            logging.info(f"{len(texts)}件のドキュメントを読み込みました。埋め込みを作成中...")
            embedding_manager.create_embeddings(texts)
            logging.info("埋め込みの作成が完了しました。")
        else:
            logging.warning("読み込めるドキュメントが見つかりませんでした。")
            return
        input_loop(answer_generator, embedding_manager)
    except Exception as e:
        logging.exception(f"致命的なエラーが発生しました: {e}")

if __name__ == "__main__":
    main()