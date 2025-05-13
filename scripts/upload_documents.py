#!/usr/bin/env python3
"""
ベクトルDB（pgvector/PostgreSQL）への文書登録スクリプト
- data/knowledge配下のテキストファイルをすべて登録
- EmbeddingManager/PGVectorを用いる
- .envやconfig.pyからDB接続情報を取得
"""
import os
from src.rag.embeddings.embedding_manager import EmbeddingManager
from langchain_postgres import PGVector
from langchain_core.documents import Document
from src.config import get_settings
from typing import List


def load_documents_from_directory(directory: str) -> List[Document]:
    """
    指定ディレクトリ配下の全テキストファイルをDocument化して返す。
    ファイルパスをmetadata['source']に格納。
    """
    docs = []
    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.endswith(".txt"):
                path = os.path.join(root, fname)
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                docs.append(Document(page_content=content, metadata={"source": path}))
    return docs


def get_vector_store() -> PGVector:
    """
    EmbeddingManagerとPGVectorを初期化し、ベクトルストアを返す。
    """
    settings = get_settings()
    embedding_manager = EmbeddingManager()
    embedding_function = embedding_manager.embedding_model
    vector_store = PGVector(
        embeddings=embedding_function,
        collection_name=settings.PGVECTOR_COLLECTION,
        connection=f"postgresql+psycopg://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}",
        use_jsonb=True,
    )
    return vector_store


def register_documents(vector_store: PGVector, docs: List[Document]) -> int:
    """
    文書リストをベクトルDBに登録し、登録件数を返す。
    """
    if not docs:
        return 0
    vector_store.add_documents(docs)
    return len(docs)


def print_result(num_docs: int) -> None:
    """
    登録結果を出力する。
    """
    if num_docs == 0:
        print("登録対象のテキストファイルが見つかりませんでした。")
    else:
        print(f"{num_docs}件の文書を登録しました。")


def main() -> None:
    """
    文書登録バッチのフロー制御のみを担当。
    """
    docs = load_documents_from_directory("data/knowledge")
    vector_store = get_vector_store()
    num_docs = register_documents(vector_store, docs)
    print_result(num_docs)


if __name__ == "__main__":
    main()