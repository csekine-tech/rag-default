from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import BedrockEmbeddings, HuggingFaceEmbeddings
from langchain_community.chat_models import BedrockChat
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from typing import Any, Dict, Optional
from src.rag.models.model_config import ModelType, EmbeddingModelConfig, GenerationModelConfig

class ModelFactory:
    def __init__(self):
        """初期化"""
        pass

    def create_chat_model(self) -> Any:
        """チャットモデルの作成"""
        # 優先順位: OpenAI > Anthropic > Vertex AI
        try:
            return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        except Exception as e:
            print(f"OpenAIモデルの初期化に失敗: {e}")
            try:
                return ChatAnthropic(model="claude-2", temperature=0)
            except Exception as e:
                print(f"Anthropicモデルの初期化に失敗: {e}")
                try:
                    return ChatVertexAI(temperature=0)
                except Exception as e:
                    print(f"Vertex AIモデルの初期化に失敗: {e}")
                    raise Exception("利用可能なチャットモデルがありません")

    def create_embedding_model(self) -> Any:
        """埋め込みモデルの作成"""
        try:
            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                dimensions=1536  # text-embedding-3-smallの次元数
            )
        except Exception as e:
            print(f"OpenAI Embeddingsの初期化に失敗: {e}")
            raise Exception("埋め込みモデルの初期化に失敗しました")

    @staticmethod
    def create_generation_model(config: GenerationModelConfig) -> Any:
        """生成モデルを作成"""
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
            raise ValueError(f"Unsupported generation model type: {config.model_type}")