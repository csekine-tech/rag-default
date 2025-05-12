import json
import os
from typing import List
from langchain.schema import Document
from src.rag.embeddings.embedding_manager import EmbeddingManager
from src.rag.retrieval.retriever import Retriever
from src.rag.utils.message_manager import MessageManager

def load_test_documents(file_path: str) -> List[Document]:
    """JSONファイルからテスト文書を読み込む"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []
    for doc in data['documents']:
        documents.append(Document(
            page_content=doc['content'],
            metadata=doc['metadata']
        ))

    return documents

def add_test_documents():
    """テスト用の文書をベクトルストアに追加"""
    # ベースディレクトリの取得
    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    # ファイルパスの設定
    test_docs_path = os.path.join(current_dir, 'data', 'evaluation', 'test_documents.json')
    messages_path = os.path.join(current_dir, 'data', 'evaluation', 'error_messages.json')

    # メッセージマネージャーの初期化
    try:
        message_manager = MessageManager(messages_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"メッセージファイルの読み込みに失敗しました: {e}")
        return

    # テスト文書の読み込み
    try:
        test_documents = load_test_documents(test_docs_path)
    except FileNotFoundError:
        print(message_manager.get_error_message('file_not_found', file_path=test_docs_path))
        return
    except json.JSONDecodeError:
        print(message_manager.get_error_message('invalid_json', file_path=test_docs_path))
        return

    # Retrieverの初期化と文書の追加
    try:
        embedding_manager = EmbeddingManager()
        retriever = Retriever(embedding_manager)
        retriever.add_documents(test_documents)
        print(message_manager.get_success_message('documents_added'))
    except Exception as e:
        print(message_manager.get_error_message('vector_store_error', error=str(e)))

if __name__ == "__main__":
    add_test_documents()