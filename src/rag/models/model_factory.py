from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.chat_models import BedrockChat
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from typing import Any, Dict, Optional
from src.rag.models.model_config import ModelType, EmbeddingModelConfig, GenerationModelConfig

class ModelFactory:
    @staticmethod
    def create_embedding_model(config: EmbeddingModelConfig) -> Any:
        """埋め込みモデルを作成"""
        if config.model_type == ModelType.OPENAI:
            return OpenAIEmbeddings(
                model=config.model_name,
                openai_api_key=config.openai_api_key
            )
        elif config.model_type == ModelType.BEDROCK:
            return BedrockEmbeddings(
                model_id=config.model_name,
                aws_access_key_id=config.aws_access_key_id,
                aws_secret_access_key=config.aws_secret_access_key
            )
        elif config.model_type == ModelType.VERTEX:
            return VertexAIEmbeddings(
                model_name=config.model_name,
                credentials_path=config.google_application_credentials
            )
        else:
            raise ValueError(f"Unsupported embedding model type: {config.model_type}")

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