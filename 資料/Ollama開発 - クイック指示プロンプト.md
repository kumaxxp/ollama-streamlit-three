# 🚀 Ollama開発 - クイック指示プロンプト

## 必須情報
- **リポジトリ**: https://github.com/kumaxxp/ollama-streamlit-chat
- **環境**: Ubuntu 24.04, RTX A5000, Python 3.11
- **Conda**: `conda activate ollama-chat`
- **起動**: `streamlit run app/simple_chat.py`

## 実装済み
✅ Streamlit + Ollamaチャット基本機能  
✅ Qwen2.5-7B, Gemma3対応  
✅ ストリーミング応答  
✅ 会話履歴管理  

## 開発ルール
1. **必ず`ollama-chat`環境を使用**
2. **新パッケージ追加時は`requirements.txt`更新**
3. **`app/`にメイン機能、`utils/`にユーティリティ**
4. **Ollamaポート: 11434**

## 次の実装候補
- キャラクター設定機能
- 会話分析ページ
- RAG実装
- 音声入出力

**このリポジトリをベースに機能追加・改善を行ってください。**