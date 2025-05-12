from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
import os
from typing import List, Optional, Callable
from src.config import get_settings, get_embedding_model_config
from src.rag.models.model_factory import ModelFactory
from langchain.schema import Document

class EmbeddingManager:
    def __init__(self):
        self.settings = get_settings()
        self.embedding_model = ModelFactory.create_embedding_model(get_embedding_model_config())
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP
        )
        self.vector_store = Chroma(
            persist_directory=self.settings.CHROMA_PERSIST_DIRECTORY,
            embedding_function=self.embedding_model
        )
        self._initialize_vector_store()

    def _initialize_vector_store(self) -> None:
        """ベクトルストアの初期化"""
        # 永続化ディレクトリが存在しない場合は作成
        os.makedirs(self.settings.CHROMA_PERSIST_DIRECTORY, exist_ok=True)

        # 既存のベクトルストアを読み込む
        try:
            self.vector_store = Chroma(
                persist_directory=self.settings.CHROMA_PERSIST_DIRECTORY,
                embedding_function=self.embedding_model
            )
        except Exception as e:
            print(f"ベクトルストアの初期化中にエラーが発生しました: {e}")
            self.vector_store = None

    def load_documents(self, directory: str) -> List[str]:
        """指定されたディレクトリからドキュメントを読み込む"""
        loader = DirectoryLoader(
            directory,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        documents = loader.load()
        return [doc.page_content for doc in documents]

    def create_embeddings(self, texts: List[str]) -> None:
        """テキストの埋め込みを作成し、ベクトルストアに保存"""
        chunks = self.text_splitter.create_documents(texts)
        self.vector_store.add_documents(chunks)

    def search_similar(self, query: str, k: int = 4) -> List[Document]:
        """類似ドキュメントの検索"""
        return self.vector_store.similarity_search(query, k=k)

    def add_documents(self, documents: List[Document]) -> None:
        """ドキュメントをベクトルストアに追加"""
        self.vector_store.add_documents(documents)
        self.vector_store.persist()

    def get_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """テキスト分割器の取得"""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP
        )

    def get_embedding_function(self) -> Callable:
        """埋め込み関数を取得"""
        return self.embedding_model.embed_query