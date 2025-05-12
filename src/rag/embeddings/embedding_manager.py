import logging
import os
from typing import List, Optional, Callable
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.schema import Document
from config import get_settings, get_embedding_model_config
from rag.models.model_factory import ModelFactory

class EmbeddingManagerError(Exception):
    """
    EmbeddingManager関連の例外クラス。
    """
    pass

class EmbeddingManager:
    """
    ドキュメントのロード、埋め込み生成、ベクトルストア管理を行うクラス。
    embedding_modelやvector_storeはDI可能。
    """
    def __init__(
        self,
        embedding_model: Optional[object] = None,
        vector_store: Optional[Chroma] = None,
        settings: Optional[object] = None
    ):
        """
        Args:
            embedding_model (Optional[object]): 埋め込みモデルインスタンス
            vector_store (Optional[Chroma]): ベクトルストアインスタンス
            settings (Optional[object]): 設定オブジェクト
        """
        self.settings = settings or get_settings()
        self.embedding_model = embedding_model or ModelFactory.create_embedding_model(get_embedding_model_config())
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP
        )
        self.vector_store = vector_store or Chroma(
            persist_directory=self.settings.CHROMA_PERSIST_DIRECTORY,
            embedding_function=self.embedding_model
        )
        self._initialize_vector_store()

    def _initialize_vector_store(self) -> None:
        """
        ベクトルストアの初期化。永続化ディレクトリの作成と既存ストアのロード。
        Raises:
            EmbeddingManagerError: 初期化失敗時
        """
        try:
            os.makedirs(self.settings.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
            self.vector_store = Chroma(
                persist_directory=self.settings.CHROMA_PERSIST_DIRECTORY,
                embedding_function=self.embedding_model
            )
        except Exception as e:
            logging.error(f"ベクトルストアの初期化中にエラー: {e}")
            raise EmbeddingManagerError(f"ベクトルストアの初期化中にエラー: {e}")

    def load_documents(self, directory: str) -> List[str]:
        """
        指定されたディレクトリからドキュメントを読み込む。
        Args:
            directory (str): ドキュメントディレクトリのパス
        Returns:
            List[str]: 読み込んだドキュメントのテキストリスト
        """
        loader = DirectoryLoader(
            directory,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        documents = loader.load()
        return [doc.page_content for doc in documents]

    def create_embeddings(self, texts: List[str]) -> None:
        """
        テキストの埋め込みを作成し、ベクトルストアに保存する。
        Args:
            texts (List[str]): 埋め込み対象テキストリスト
        """
        chunks = self.text_splitter.create_documents(texts)
        self.vector_store.add_documents(chunks)

    def search_similar(self, query: str, k: int = 4) -> List[Document]:
        """
        類似ドキュメントの検索。
        Args:
            query (str): クエリテキスト
            k (int): 取得件数
        Returns:
            List[Document]: 類似ドキュメントリスト
        """
        return self.vector_store.similarity_search(query, k=k)

    def add_documents(self, documents: List[Document]) -> None:
        """
        ドキュメントをベクトルストアに追加し、永続化する。
        Args:
            documents (List[Document]): 追加するドキュメント
        """
        self.vector_store.add_documents(documents)
        self.vector_store.persist()

    def get_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """
        テキスト分割器の取得。
        Returns:
            RecursiveCharacterTextSplitter: テキスト分割器
        """
        return RecursiveCharacterTextSplitter(
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP
        )

    def get_embedding_function(self) -> Callable:
        """
        埋め込み関数を取得。
        Returns:
            Callable: 埋め込み関数
        """
        return self.embedding_model.embed_query

# TODO: 設定値や入出力の型安全性向上のため、pydantic/dataclassの活用を検討