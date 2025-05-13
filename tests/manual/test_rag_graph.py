import logging
logging.basicConfig(level=logging.INFO)

from src.rag.graph import build_rag_graph


def main() -> None:
    """
    LangGraphベースRAGワークフロー（retriever→llm）の動作確認用スクリプト。
    .env, PostgreSQL, OpenAI APIキー、ベクトルDBの事前構築が必要です。
    """
    graph = build_rag_graph()

    # テスト用クエリ
    query = "RAGの利点は？"

    # LangGraphのワークフローを実行
    result = graph.invoke({"query": query})

    print("=== 検索結果 ===")
    docs = result.get("retrieved_docs")
    if docs:
        for doc in docs:
            print(doc.page_content)
    else:
        print("（検索結果なし）")

    print("=== LLM回答 ===")
    print(result.get("generated_answer"))


if __name__ == "__main__":
    main()