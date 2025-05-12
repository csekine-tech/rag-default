import os
import shutil
import pytest
from unittest.mock import MagicMock
from rag.embeddings.embedding_manager import EmbeddingManager, EmbeddingManagerError
from langchain.schema import Document

@pytest.fixture
def temp_docs_dir(tmp_path):
    """
    テスト用の一時ドキュメントディレクトリを作成するfixture。
    """
    dir_path = tmp_path / "docs"
    dir_path.mkdir()
    # テスト用テキストファイルを2つ作成
    file1 = dir_path / "doc1.txt"
    file2 = dir_path / "doc2.txt"
    file1.write_text("これはテストドキュメント1です。", encoding="utf-8")
    file2.write_text("これはテストドキュメント2です。", encoding="utf-8")
    yield str(dir_path)
    # 後始末（pytestのtmp_pathは自動削除されるが念のため）
    shutil.rmtree(str(dir_path), ignore_errors=True)

def test_load_documents(temp_docs_dir):
    """
    EmbeddingManager.load_documentsがディレクトリ内のテキストファイルを正しく読み込むことを検証する。
    """
    manager = EmbeddingManager()
    docs = manager.load_documents(temp_docs_dir)
    # 2ファイル分のテキストが取得できること
    assert len(docs) == 2
    assert "テストドキュメント1" in docs[0] or "テストドキュメント1" in docs[1]
    assert "テストドキュメント2" in docs[0] or "テストドキュメント2" in docs[1]

@pytest.fixture
def mock_embedding_manager():
    """
    Chromaやembedding_modelをモック化したEmbeddingManagerを返すfixture。
    """
    mock_embedding_model = MagicMock()
    # 1536次元のベクトルを返す
    mock_embedding_model.embed_documents.return_value = [
        [0.0] * 1536, [0.0] * 1536
    ]
    mock_embedding_model.embed_query.return_value = [0.0] * 1536
    mock_vector_store = MagicMock()
    # similarity_searchはDocumentのリストを返すようにする
    mock_vector_store.similarity_search.return_value = [
        Document(page_content="モックドキュメント1"),
        Document(page_content="モックドキュメント2")
    ]
    manager = EmbeddingManager(
        embedding_model=mock_embedding_model,
        vector_store=mock_vector_store
    )
    # _initialize_vector_storeで上書きされるのを防ぐため、明示的に再代入
    manager.vector_store = mock_vector_store
    return manager, mock_vector_store

def test_create_embeddings_calls_add_documents(mock_embedding_manager):
    """
    create_embeddingsがvector_store.add_documentsを呼び出すことを検証する。
    """
    manager, mock_vector_store = mock_embedding_manager
    # text_splitter.create_documentsもモック化
    manager.text_splitter.create_documents = MagicMock(return_value=[
        Document(page_content="chunk1"),
        Document(page_content="chunk2")
    ])
    manager.create_embeddings(["テスト1", "テスト2"])
    manager.text_splitter.create_documents.assert_called_once()
    mock_vector_store.add_documents.assert_called_once_with([
        Document(page_content="chunk1"),
        Document(page_content="chunk2")
    ])

def test_search_similar_returns_documents(mock_embedding_manager):
    """
    search_similarがvector_store.similarity_searchを呼び出し、Documentリストを返すことを検証する。
    """
    manager, _ = mock_embedding_manager
    docs = manager.search_similar("質問")
    assert isinstance(docs, list)
    assert all(isinstance(doc, Document) for doc in docs)
    assert "モックドキュメント1" in [doc.page_content for doc in docs]

def test_add_documents_calls_add_and_persist(mock_embedding_manager):
    """
    add_documentsがvector_store.add_documentsとpersistを呼び出すことを検証する。
    """
    manager, mock_vector_store = mock_embedding_manager
    docs = [Document(page_content="追加ドキュメント")]
    manager.add_documents(docs)
    mock_vector_store.add_documents.assert_called_once_with(docs)
    mock_vector_store.persist.assert_called_once()

def test_initialize_vector_store_error(monkeypatch):
    """
    _initialize_vector_storeで例外が発生した場合にEmbeddingManagerErrorが送出されることを検証する。
    """
    # os.makedirsで例外を発生させる
    monkeypatch.setattr(os, "makedirs", MagicMock(side_effect=OSError("ディレクトリ作成失敗")))
    with pytest.raises(EmbeddingManagerError):
        EmbeddingManager(settings=None)