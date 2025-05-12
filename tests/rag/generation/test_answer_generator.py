import pytest
from unittest.mock import MagicMock
from rag.generation.answer_generator import AnswerGenerator
import tempfile
import os

@pytest.fixture
def temp_prompt_file():
    """
    テスト用の一時プロンプトテンプレートファイルを作成するfixture。
    """
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as f:
        f.write("質問: {question}\nコンテキスト: {context}\n回答してください。")
        path = f.name
    yield path
    os.remove(path)

def test_generate_returns_llm_response(temp_prompt_file):
    """
    generateがLLMの出力を正しく返すことを検証する。
    """
    # LLMをモック化
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "これはモック回答です。"
    # chain.invokeでモックレスポンスを返す
    def fake_chain_invoke(inputs):
        return mock_response
    # ChatPromptTemplate.from_template().__or__()でchainを返すようにする
    mock_prompt = MagicMock()
    mock_prompt.__or__.return_value.invoke = fake_chain_invoke
    # AnswerGeneratorのインスタンス生成
    generator = AnswerGenerator(prompt_template_path=temp_prompt_file, llm=mock_llm)
    # promptを直接モックに差し替え
    generator.prompt = mock_prompt
    answer = generator.generate("テスト質問", "テストコンテキスト")
    assert answer == "これはモック回答です。"

def test_generate_handles_llm_exception(temp_prompt_file):
    """
    generateでLLMが例外を投げた場合にエラーメッセージを返すことを検証する。
    """
    mock_llm = MagicMock()
    mock_prompt = MagicMock()
    # chain.invokeで例外を発生させる
    def raise_exception(inputs):
        raise RuntimeError("LLMエラー")
    mock_prompt.__or__.return_value.invoke = raise_exception
    generator = AnswerGenerator(prompt_template_path=temp_prompt_file, llm=mock_llm)
    generator.prompt = mock_prompt
    answer = generator.generate("テスト質問", "テストコンテキスト")
    assert "エラー" in answer or "error" in answer.lower()