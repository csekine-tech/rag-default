from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

class AnswerGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたは高性能な質問応答システムです。
以下のガイドラインに厳密に従って回答を生成してください：

1. 与えられたコンテキストの情報「のみ」を使用してください
2. コンテキストに明示的に含まれていない情報は、たとえ正しいと思われても「絶対に」使用しないでください
3. 質問に直接関連する情報のみを含めてください
4. 簡潔で明確な回答を心がけてください
5. 質問の意図を正確に理解し、それに応じた回答を提供してください
6. コンテキストの情報を言い換える場合も、元の意味を厳密に保持してください
7. コンテキストにない情報を推測や一般化で補完しないでください

回答の各部分が、必ずコンテキストの特定の部分に基づいていることを確認してください。"""),
            ("human", """質問: {question}

利用可能なコンテキスト:
{context}

上記のコンテキストのみを使用して、質問に対する回答を生成してください。
コンテキストに含まれていない情報は一切含めないでください。""")
        ])

    def generate(self, question: str, context: str) -> str:
        """質問とコンテキストから回答を生成"""
        chain = self.prompt | self.llm
        response = chain.invoke({
            "question": question,
            "context": context
        })
        return response.content