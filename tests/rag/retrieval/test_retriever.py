import pytest
from unittest.mock import MagicMock
from rag.retrieval.retriever import Retriever, RetrieverError
from langchain.schema import Document

@pytest.fixture
def mock_vectorstore():
    """
    similarity_search_with_score, add_documents, persistをモック化したvectorstoreを返すfixture。
    """
    mock_vs = MagicMock()
    # similarity_search_with_scoreは(doc, score)のリストを返す
    mock_vs.similarity_search_with_score.return_value = [
        (Document(page_content="ドキュメント1"), 0.8),
        (Document(page_content="ドキュメント2"), 0.6)
    ]
    return mock_vs

def test_add_documents_calls_vectorstore_methods(mock_vectorstore):
    """
    add_documentsがvectorstore.add_documentsとpersistを呼び出すことを検証する。
    """
    retriever = Retriever(vectorstore=mock_vectorstore)
    docs = [Document(page_content="追加ドキュメント")]
    retriever.add_documents(docs)
    mock_vectorstore.add_documents.assert_called_once_with(docs)
    mock_vectorstore.persist.assert_called_once()

def test_add_documents_raises_on_error(mock_vectorstore):
    """
    add_documentsでvectorstore.add_documentsが例外を投げた場合にRetrieverErrorが送出されることを検証する。
    """
    retriever = Retriever(vectorstore=mock_vectorstore)
    mock_vectorstore.add_documents.side_effect = RuntimeError("追加失敗")
    with pytest.raises(RetrieverError):
        retriever.add_documents([Document(page_content="失敗ドキュメント")])

def test_retrieve_filters_by_score_and_returns_context(mock_vectorstore):
    """
    retrieveがスコア閾値でフィルタし、正しいコンテキストを返すことを検証する。
    """
    retriever = Retriever(vectorstore=mock_vectorstore)
    # 0.7以上のみ返す
    context = retriever.retrieve("質問", top_k=3, similarity_threshold=0.7)
    assert "ドキュメント1" in context
    assert "ドキュメント2" not in context

def test_retrieve_returns_no_info_message_when_no_docs(mock_vectorstore):
    """
    retrieveで閾値を上げて該当文書がない場合にメッセージを返すことを検証する。
    """
    retriever = Retriever(vectorstore=mock_vectorstore)
    # 0.9以上にすると全て除外される
    context = retriever.retrieve("質問", top_k=3, similarity_threshold=0.9)
    assert "関連する情報が見つかりません" in context

def test_retrieve_raises_on_error(mock_vectorstore):
    """
    retrieveでvectorstore.similarity_search_with_scoreが例外を投げた場合にRetrieverErrorが送出されることを検証する。
    """
    retriever = Retriever(vectorstore=mock_vectorstore)
    mock_vectorstore.similarity_search_with_score.side_effect = RuntimeError("検索失敗")
    with pytest.raises(RetrieverError):
        retriever.retrieve("質問")