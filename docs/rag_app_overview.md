# RAGチャットアプリ 実装・運用ドキュメント

## 1. プロジェクト概要
- Python製の最新RAG（Retrieval-Augmented Generation）チャットアプリ。
- UIレス、CLIベースで日本語ナレッジベースQAと自動評価（RAGAS）に対応。
- モデル・パイプライン切り替えや拡張性を重視した設計。
- Docker/Docker Compose対応。

## 2. ディレクトリ構成（抜粋）
```
.
├── data/
│   ├── knowledge/         # ナレッジベース（AI, LLM, RAG等の日本語テキスト）
│   └── evaluation/        # RAGAS評価用データセット
├── docs/                  # ドキュメント
├── src/
│   ├── config.py          # 環境変数・設定管理
│   ├── main.py            # ナレッジベース構築・対話QAエントリ
│   ├── evaluate.py        # RAGAS自動評価
│   └── rag/
│       ├── embeddings/    # ベクトルDB・埋め込み管理
│       ├── generation/    # 解答生成
│       ├── retrieval/     # ベクトル検索
│       ├── evaluation/    # RAGAS評価
│       └── models/        # モデル抽象化
└── docker-compose.yml
```

## 3. 主な実装内容
- **設定管理**: `src/config.py`でpydantic+dotenvによる環境変数管理。
- **埋め込み・ベクトルDB**: `src/rag/embeddings/embedding_manager.py`でChroma永続化・分割・登録。
- **検索**: `src/rag/retrieval/retriever.py`で類似検索・しきい値フィルタ。
- **解答生成**: `src/rag/generation/answer_generator.py`でLLM（OpenAI等）による厳密なコンテキストQA。
- **モデル切替**: `src/rag/models/model_factory.py`等でOpenAI/Anthropic/Bedrock/Vertex AI等に対応。
- **評価**: `src/rag/evaluation/rag_evaluator.py`および`src/evaluate.py`でRAGAS指標による自動評価。
- **テスト**: `data/evaluation/test_dataset.py`にサンプル質問・正解セット。

## 4. データ概要
### ナレッジベース（`data/knowledge/`）
- `ai_overview.txt` : AIの概要
- `llm_technology.txt` : LLM技術の解説
- `rag_system.txt` : RAGシステムの説明
- `sample.txt` : サンプルテキスト

### 評価データ（`data/evaluation/test_dataset.py`）
- RAGAS評価用の日本語質問・正解セット（例: RAGの利点、LLMの種類、AI応用分野など）

## 5. 実行方法
### 事前準備
- `.env`ファイルにAPIキー等を設定
- 必要パッケージは`requirements.txt`に記載
- Docker環境推奨

### ナレッジベース構築・QA
```sh
# 初回のみ: ベクトルDB永続化ディレクトリを削除して再構築推奨
rm -rf data/chroma
# ナレッジベースの埋め込み・対話QA
docker-compose run --rm rag-app python -m src.main
```

### RAGASによる自動評価
```sh
docker-compose run --rm rag-app python -m src.evaluate
```

## 6. 評価方法（RAGAS）
- `src/evaluate.py`でRAGAS指標（faithfulness, answer_relevancy, context_precision, context_recall）を自動計算。
- 結果は指標ごとにスコアまたは詳細dictで出力。

## 7. 注意事項・トラブルシュート
- Chromaやtiktokenの初回利用時はインターネット接続必須。
- ベクトルDBスキーマエラー時は`data/chroma`を削除して再構築。
- APIキーやパスの大文字・小文字、.envの設定ミスに注意。
- LangChain/Chroma/ragas等のバージョン競合時は`requirements.txt`を調整。

---

ご不明点や拡張要望は`docs/`配下に追記してください。