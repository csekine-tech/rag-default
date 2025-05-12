import json
import os
from typing import Dict, Any

class MessageManager:
    def __init__(self, messages_file_path: str):
        """メッセージマネージャーの初期化"""
        self.messages = self._load_messages(messages_file_path)

    def _load_messages(self, file_path: str) -> Dict[str, Any]:
        """メッセージファイルを読み込む"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"メッセージファイルが見つかりません: {file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"メッセージファイルのJSON形式が不正です: {file_path}")

    def get_error_message(self, error_key: str, **kwargs) -> str:
        """エラーメッセージを取得"""
        try:
            error = self.messages['errors'][error_key]
            return error['message'].format(**kwargs)
        except KeyError:
            return f"不明なエラーキー: {error_key}"

    def get_success_message(self, success_key: str, **kwargs) -> str:
        """成功メッセージを取得"""
        try:
            success = self.messages['success'][success_key]
            return success['message'].format(**kwargs)
        except KeyError:
            return f"不明な成功キー: {success_key}"