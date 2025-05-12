import pytest
from unittest.mock import patch, MagicMock
from rag.models.model_factory import ModelFactory, ModelFactoryError
from rag.models.model_config import GenerationModelConfig, ModelType
from pydantic import ValidationError

def test_create_chat_model_openai_success():
    """
    create_chat_modelがOpenAIモデルを返す場合の正常系。
    """
    with patch("rag.models.model_factory.ChatOpenAI", return_value="openai_model") as mock_openai:
        factory = ModelFactory()
        model = factory.create_chat_model()
        assert model == "openai_model"
        mock_openai.assert_called_once()

def test_create_chat_model_fallback_to_anthropic():
    """
    OpenAI失敗→Anthropic成功時にAnthropicモデルを返す。
    """
    with patch("rag.models.model_factory.ChatOpenAI", side_effect=Exception("fail")), \
         patch("rag.models.model_factory.ChatAnthropic", return_value="anthropic_model") as mock_anthropic:
        factory = ModelFactory()
        model = factory.create_chat_model()
        assert model == "anthropic_model"
        mock_anthropic.assert_called_once()

def test_create_chat_model_fallback_to_vertex():
    """
    OpenAI/Anthropic失敗→Vertex成功時にVertexモデルを返す。
    """
    with patch("rag.models.model_factory.ChatOpenAI", side_effect=Exception("fail")), \
         patch("rag.models.model_factory.ChatAnthropic", side_effect=Exception("fail")), \
         patch("rag.models.model_factory.ChatVertexAI", return_value="vertex_model") as mock_vertex:
        factory = ModelFactory()
        model = factory.create_chat_model()
        assert model == "vertex_model"
        mock_vertex.assert_called_once()

def test_create_chat_model_all_fail():
    """
    すべてのモデル初期化が失敗した場合にModelFactoryErrorが送出される。
    """
    with patch("rag.models.model_factory.ChatOpenAI", side_effect=Exception("fail")), \
         patch("rag.models.model_factory.ChatAnthropic", side_effect=Exception("fail")), \
         patch("rag.models.model_factory.ChatVertexAI", side_effect=Exception("fail")):
        factory = ModelFactory()
        with pytest.raises(ModelFactoryError):
            factory.create_chat_model()

def test_create_embedding_model_success():
    """
    create_embedding_modelがOpenAIEmbeddingsを返す場合の正常系。
    """
    with patch("rag.models.model_factory.OpenAIEmbeddings", return_value="embeddings_model") as mock_emb:
        factory = ModelFactory()
        model = factory.create_embedding_model()
        assert model == "embeddings_model"
        mock_emb.assert_called_once()

def test_create_embedding_model_fail():
    """
    create_embedding_modelで初期化失敗時にModelFactoryErrorが送出される。
    """
    with patch("rag.models.model_factory.OpenAIEmbeddings", side_effect=Exception("fail")):
        factory = ModelFactory()
        with pytest.raises(ModelFactoryError):
            factory.create_embedding_model()

def test_create_generation_model_openai():
    """
    create_generation_modelでOpenAIモデルが返る。
    """
    config = GenerationModelConfig(model_type=ModelType.OPENAI, model_name="gpt", openai_api_key="key")
    with patch("rag.models.model_factory.ChatOpenAI", return_value="openai_model") as mock_openai:
        model = ModelFactory.create_generation_model(config)
        assert model == "openai_model"
        mock_openai.assert_called_once_with(model="gpt", openai_api_key="key")

def test_create_generation_model_unsupported():
    """
    create_generation_modelで未対応モデル種別時にValidationErrorが送出されることを検証する。
    """
    with pytest.raises(ValidationError):
        GenerationModelConfig(model_type="UNSUPPORTED", model_name="xxx")