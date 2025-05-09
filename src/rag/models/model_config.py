from enum import Enum
from typing import Optional
from pydantic import BaseModel, ConfigDict

class ModelType(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    VERTEX = "vertex"

class EmbeddingModelConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_type: ModelType
    model_name: str
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    google_application_credentials: Optional[str] = None

class GenerationModelConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_type: ModelType
    model_name: str
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    google_application_credentials: Optional[str] = None

# デフォルトのモデル設定
DEFAULT_EMBEDDING_MODEL = EmbeddingModelConfig(
    model_type=ModelType.OPENAI,
    model_name="text-embedding-3-small",
    openai_api_key="",
    anthropic_api_key=None,
    aws_access_key_id=None,
    aws_secret_access_key=None,
    google_application_credentials=None
)

DEFAULT_GENERATION_MODEL = GenerationModelConfig(
    model_type=ModelType.OPENAI,
    model_name="gpt-3.5-turbo",
    openai_api_key="",
    anthropic_api_key=None,
    aws_access_key_id=None,
    aws_secret_access_key=None,
    google_application_credentials=None
)