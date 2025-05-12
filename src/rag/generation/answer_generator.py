import logging
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os

class AnswerGenerator:
    """
    質問とコンテキストからLLMを用いて回答を生成するクラス。
    プロンプトテンプレートは外部ファイルから読み込む。
    LLMやプロンプトは依存性注入可能。
    """
    def __init__(
        self,
        prompt_template_path: str = "prompt_templates/answer_generator_prompt.txt",
        llm: Optional[ChatOpenAI] = None
    ):
        """
        Args:
            prompt_template_path (str): プロンプトテンプレートファイルのパス
            llm (Optional[ChatOpenAI]): LLMインスタンス（未指定時はデフォルトを使用）
        """
        self.prompt_template_path = prompt_template_path
        self.llm = llm or ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0
        )
        self.prompt = self._load_prompt()

    def _load_prompt(self) -> ChatPromptTemplate:
        """
        プロンプトテンプレートファイルを読み込み、ChatPromptTemplateを生成する。
        Returns:
            ChatPromptTemplate: プロンプトテンプレート
        Raises:
            FileNotFoundError: テンプレートファイルが存在しない場合
        """
        if not os.path.exists(self.prompt_template_path):
            logging.error(f"プロンプトテンプレートが見つかりません: {self.prompt_template_path}")
            raise FileNotFoundError(f"プロンプトテンプレートが見つかりません: {self.prompt_template_path}")
        with open(self.prompt_template_path, "r", encoding="utf-8") as f:
            template = f.read()
        return ChatPromptTemplate.from_template(template)

    def generate(self, question: str, context: str) -> str:
        """
        質問とコンテキストから回答を生成する。
        Args:
            question (str): 質問文
            context (str): コンテキスト情報
        Returns:
            str: 生成された回答
        """
        try:
            chain = self.prompt | self.llm
            response = chain.invoke({
                "question": question,
                "context": context
            })
            return response.content
        except Exception as e:
            logging.error(f"LLMによる回答生成中にエラー: {e}")
            return "回答生成中にエラーが発生しました。"