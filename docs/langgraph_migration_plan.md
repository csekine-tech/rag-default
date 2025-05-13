# 実装進捗

- 2024/06/09: 実装計画を作成・保存。LangGraphワークフロー雛形（rag/graph.py）作成準備中。
- 2024/06/09: rag/graph.py にLangGraphワークフロー雛形を実装。
- 2024/06/09: pgvector対応retriever（src/rag/retrieval/pgvector.py）雛形を実装。
- 2024/06/09: pgvector対応retrieverをLangGraphワークフロー（src/rag/graph.py）に組み込み。
- 2024/06/09: EmbeddingManagerからembedding_functionをDIしPGVectorに組み込む対応を実施。
- 2024/06/09: LLMノードをAnswerGenerator本実装に差し替え。
- 2024/06/XX: ベクトルDB文書登録スクリプト（scripts/upload_documents.py）をSRP準拠で実装・運用開始。

# LangGraph & LangSmith 移行実装計画

## 1. LangGraph導入・設計

### 1-1. LangGraphの理解と要件整理
- LangGraphはLangChainのワークフロー自動化・状態遷移管理のためのフレームワークです。
- 既存のRAGフロー（例：クエリ受信→ベクトル検索→LLM生成→レスポンス返却）をLangGraphのノード・エッジとして再設計します。

### 1-2. LangGraphノード設計
- **Retrieverノード**：ベクトルDB検索（retriever/）
- **Rerankノード**：検索結果の再ランキング（必要に応じて）
- **LLMノード**：LLM呼び出し（llm/）
- **Postprocessノード**：レスポンス整形・検証（rag/）
- **評価ノード**：メトリクス計算用（rag/metrics.py など）

### 1-3. LangGraphワークフロー定義
- 各ノードをLangGraphの`Node`として実装し、状態遷移（エッジ）を定義
- 例：`query` → `retrieve` → `rerank` → `generate` → `postprocess` → `metrics`

---

## 2. LangSmithによるメトリクス取得

### 2-1. LangSmithのセットアップ
- LangSmithのAPIキー・プロジェクト設定を`config/`で管理
- LangSmith SDKをインストールし、LangGraphの各ノードでトレース・メトリクス送信を有効化

### 2-2. メトリクスの自動記録
- LangGraphのワークフロー全体をLangSmithでトレース
- 主要なメトリクス（MRR, F1, EMなど）は`rag/metrics.py`で計算し、LangSmithに送信

---

## 3. コード構成・リファクタリング

### 3-1. ディレクトリ構成
```
.
├── llm/           # LLM呼び出し
├── retriever/     # ベクトル検索
├── rag/           # LangGraphワークフロー・評価
├── config/        # 設定
├── utils/         # 補助関数
├── tests/         # テスト
└── prompts/       # プロンプトテンプレート
```

### 3-2. 主要ファイル
- `rag/graph.py`：LangGraphワークフロー定義
- `rag/metrics.py`：評価指標の実装
- `config/langsmith.py`：LangSmith設定
- `tests/`：LangGraphワークフローのテスト

---

## 4. 実装・テスト・CI

### 4-1. 実装
- 既存処理をLangGraphノードとして分割・再実装
- 各ノードで型アノテーション・docstring・ロギングを徹底
- LangSmith連携を組み込む

### 4-2. テスト
- pytest + mockで各ノード・ワークフローの単体/統合テスト
- LangSmithのテスト用プロジェクトでメトリクス送信を検証

### 4-3. CI
- Black, Ruff, Mypy, pytest, coverageをCIに組み込む

---

## 5. 移行・運用

- 既存APIやCLIのインターフェースは維持しつつ、内部処理をLangGraphベースに切り替え
- LangSmithダッシュボードでメトリクス・トレースを可視化し、運用・改善サイクルを回す

## 6. ベクトルDBへの文書登録・運用

### 6-1. 文書登録スクリプトの実装
- `scripts/upload_documents.py`等で、EmbeddingManager/PGVectorを用いた文書アップロード処理を新規実装する。
- 例：`vector_store.add_documents([...])` で知識ベースを構築。
- 登録対象はテキスト、PDF、Webページ等を想定。

### 6-2. 運用・テストフロー
- 本番・テスト環境ごとに登録データを分離管理。
- CI/CDや初期セットアップ時に自動実行できるようにする。
- サンプルデータや運用手順もREADME等で明示。