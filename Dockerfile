FROM python:3.11-slim

WORKDIR /app

# ビルドに必要なツールとPostgreSQLクライアントライブラリのインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 必要なパッケージのインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードのコピー
COPY . .

# 環境変数の設定
ENV PYTHONPATH=/app

# データディレクトリの作成
RUN mkdir -p /app/data/chroma /app/data/knowledge

# 実行ユーザーの設定
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

CMD ["python", "-m", "main"]