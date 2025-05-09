from pydantic_settings import BaseSettings
from typing import Optional
from dotenv import load_dotenv
import os
from functools import lru_cache
from src.rag.models.model_config import ModelType, EmbeddingModelConfig, GenerationModelConfig

load_dotenv()

class Settings(BaseSettings):
    # モデル設定
    EMBEDDING_MODEL_TYPE: str = "openai"
    EMBEDDING_MODEL_NAME: str = "text-embedding-3-small"
    GENERATION_MODEL_TYPE: str = "openai"
    GENERATION_MODEL_NAME: str = "gpt-3.5-turbo"

    # APIキー
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None

    # その他の設定
    CHROMA_PERSIST_DIRECTORY: str = "data/chroma"
    KNOWLEDGE_BASE_DIRECTORY: str = "data/knowledge"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    class Config:
        env_file = ".env"
        case_sensitive = True

    def get_embedding_model_config(self) -> EmbeddingModelConfig:
        return EmbeddingModelConfig(
            model_type=ModelType(self.EMBEDDING_MODEL_TYPE.lower()),
            model_name=self.EMBEDDING_MODEL_NAME,
            openai_api_key=self.OPENAI_API_KEY,
            anthropic_api_key=self.ANTHROPIC_API_KEY,
            aws_access_key_id=self.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.AWS_SECRET_ACCESS_KEY,
            google_application_credentials=self.GOOGLE_APPLICATION_CREDENTIALS
        )

    def get_generation_model_config(self) -> GenerationModelConfig:
        return GenerationModelConfig(
            model_type=ModelType(self.GENERATION_MODEL_TYPE.lower()),
            model_name=self.GENERATION_MODEL_NAME,
            openai_api_key=self.OPENAI_API_KEY,
            anthropic_api_key=self.ANTHROPIC_API_KEY,
            aws_access_key_id=self.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.AWS_SECRET_ACCESS_KEY,
            google_application_credentials=self.GOOGLE_APPLICATION_CREDENTIALS
        )

@lru_cache()
def get_settings() -> Settings:
    return Settings()

def get_embedding_model_config() -> EmbeddingModelConfig:
    settings = get_settings()
    return settings.get_embedding_model_config()

def get_generation_model_config() -> GenerationModelConfig:
    settings = get_settings()
    return settings.get_generation_model_config()