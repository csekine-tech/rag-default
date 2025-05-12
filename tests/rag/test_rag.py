import pytest
from unittest.mock import MagicMock
from rag.rag import RAG, RAGError

@pytest.fixture
def mock_components():
    """
    EmbeddingManager, AnswerGenerator, Retrieverをモック化して返すfixture。
    """
    mock_embedding_manager = MagicMock()
    mock_answer_generator = MagicMock()
    mock_retriever = MagicMock()
    return mock_embedding_manager, mock_answer_generator, mock_retriever

def test_query_returns_expected_dict(mock_components):
    """
    queryが正常にanswer, contextを含む辞書を返すことを検証する。
    """
    _, mock_answer_generator, mock_retriever = mock_components
    mock_retriever.retrieve.return_value = "モックコンテキスト"
    mock_answer_generator.generate.return_value = "モック回答"
    rag = RAG(answer_generator=mock_answer_generator, retriever=mock_retriever)
    result = rag.query("テスト質問")
    assert result["answer"] == "モック回答"
    assert result["context"] == "モックコンテキスト"
    mock_retriever.retrieve.assert_called_once_with("テスト質問")
    mock_answer_generator.generate.assert_called_once_with("テスト質問", "モックコンテキスト")

def test_query_raises_ragerror_on_exception(mock_components):
    """
    queryでRetrieverやAnswerGeneratorが例外を投げた場合にRAGErrorが送出されることを検証する。
    """
    _, mock_answer_generator, mock_retriever = mock_components
    mock_retriever.retrieve.side_effect = RuntimeError("検索失敗")
    rag = RAG(answer_generator=mock_answer_generator, retriever=mock_retriever)
    with pytest.raises(RAGError):
        rag.query("テスト質問")