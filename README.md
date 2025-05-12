# RAGチャットアプリ 環境構築ガイド

## 概要
本プロジェクトは、Python製のRAG（Retrieval-Augmented Generation）チャットアプリです。日本語ナレッジベースQAとRAGASによる自動評価に対応し、Docker環境で簡単に構築・実行できます。

---

## 1. 必要要件
- Docker / Docker Compose
- （推奨）Python 3.11+（ローカル実行の場合）
- OpenAI等のAPIキー

---

## 2. .envファイルの作成
プロジェクトルート（rag-01/）で、サンプルの`.env.example`をコピーして`.env`を作成し、必要なAPIキーや設定値を記入してください。

```sh
cp .env.example .env
```

`.env`内の例:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
GOOGLE_APPLICATION_CREDENTIALS=...
```

---

## 3. Dockerイメージのビルド
```sh
docker-compose build
```

---

## 4. 必要パッケージの確認（ローカル実行の場合）
```sh
pip install -r requirements.txt
```

---

## 5. ナレッジベースの準備
`data/knowledge/`配下に日本語テキスト（例: ai_overview.txt, llm_technology.txt, rag_system.txt）を配置してください。

---

## 6. 初回セットアップ（推奨）
ベクトルDBのスキーマ不整合を防ぐため、初回は永続化ディレクトリを削除してから実行してください。
```sh
rm -rf data/chroma
```

---

## 7. ナレッジベース構築・QAの実行
```sh
docker-compose run --rm rag-app python -m src.main
```

---

## 8. RAGASによる自動評価
```sh
docker-compose run --rm --remove-orphans rag-app python -m src.evaluate
```

---

## 9. トラブルシュート
- Chromaやtiktokenの初回利用時はインターネット接続が必要です。
- APIキーやパスの大文字・小文字、.envの設定ミスに注意してください。
- ベクトルDBエラー時は`data/chroma`を削除して再構築してください。
- 依存パッケージのバージョン競合時は`requirements.txt`を調整してください。

---

## 参考
- 詳細な実装・運用・データ概要は`docs/rag_app_overview.md`を参照してください。