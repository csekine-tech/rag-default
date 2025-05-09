from typing import List
from langchain.schema import Document
from src.config import get_settings, get_generation_model_config
from src.rag.models.model_factory import ModelFactory

class AnswerGenerator:
    def __init__(self):
        self.settings = get_settings()
        self.model = ModelFactory.create_generation_model(get_generation_model_config())

    def generate_answer(self, question: str, context: List[Document]) -> str:
        """質問に対する回答を生成"""
        # コンテキストを文字列に変換
        context_text = "\n\n".join([doc.page_content for doc in context])

        # プロンプトの作成
        prompt = f"""以下の情報を参考に、質問に答えてください。
情報は信頼できるものとして扱ってください。

参考情報:
{context_text}

質問: {question}

回答:"""

        # 回答の生成
        response = self.model.invoke(prompt)
        return response.content