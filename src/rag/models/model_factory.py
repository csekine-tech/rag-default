import logging
from typing import Any
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import BedrockEmbeddings, HuggingFaceEmbeddings
from langchain_community.chat_models import BedrockChat
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from rag.models.model_config import ModelType, EmbeddingModelConfig, GenerationModelConfig

class ModelFactoryError(Exception):
    """
    ModelFactory関連の例外クラス。
    """
    pass

class ModelFactory:
    """
    LLM・埋め込みモデルの生成を担うファクトリクラス。
    各種プロバイダ・設定に応じてインスタンスを生成。
    """
    def __init__(self):
        """
        初期化
        """
        pass

    def create_chat_model(self) -> Any:
        """
        チャットモデルの作成。
        優先順位: OpenAI > Anthropic > Vertex AI
        Returns:
            Any: チャットモデルインスタンス
        Raises:
            ModelFactoryError: いずれのモデルも初期化できない場合
        """
        try:
            return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        except Exception as e:
            logging.warning(f"OpenAIモデルの初期化に失敗: {e}")
            try:
                return ChatAnthropic(model="claude-2", temperature=0)
            except Exception as e:
                logging.warning(f"Anthropicモデルの初期化に失敗: {e}")
                try:
                    return ChatVertexAI(temperature=0)
                except Exception as e:
                    logging.error(f"Vertex AIモデルの初期化に失敗: {e}")
                    raise ModelFactoryError("利用可能なチャットモデルがありません")

    def create_embedding_model(self) -> Any:
        """
        埋め込みモデルの作成。
        Returns:
            Any: 埋め込みモデルインスタンス
        Raises:
            ModelFactoryError: 初期化失敗時
        """
        try:
            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                dimensions=1536
            )
        except Exception as e:
            logging.error(f"OpenAI Embeddingsの初期化に失敗: {e}")
            raise ModelFactoryError("埋め込みモデルの初期化に失敗しました")

    @staticmethod
    def create_generation_model(config: GenerationModelConfig) -> Any:
        """
        生成モデルを作成。
        Args:
            config (GenerationModelConfig): モデル設定
        Returns:
            Any: 生成モデルインスタンス
        Raises:
            ModelFactoryError: 未対応のモデル種別
        """
        if config.model_type == ModelType.OPENAI:
            return ChatOpenAI(
                model=config.model_name,
                openai_api_key=config.openai_api_key
            )
        elif config.model_type == ModelType.ANTHROPIC:
            return ChatAnthropic(
                model=config.model_name,
                anthropic_api_key=config.anthropic_api_key
            )
        elif config.model_type == ModelType.BEDROCK:
            return BedrockChat(
                model_id=config.model_name,
                aws_access_key_id=config.aws_access_key_id,
                aws_secret_access_key=config.aws_secret_access_key
            )
        elif config.model_type == ModelType.VERTEX:
            return ChatVertexAI(
                model_name=config.model_name,
                credentials_path=config.google_application_credentials
            )
        else:
            logging.error(f"Unsupported generation model type: {config.model_type}")
            raise ModelFactoryError(f"Unsupported generation model type: {config.model_type}")

# TODO: 設定値や入出力の型安全性向上のため、pydantic/dataclassの活用を検討