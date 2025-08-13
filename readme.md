# 🤖 Ollama Streamlit Chat

ローカルLLMと対話するための高機能WebUIチャットシステム。Ollama + Streamlitで構築された、拡張可能なAIチャットプラットフォームです。

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.32.0-red.svg)
![Ollama](https://img.shields.io/badge/ollama-0.3.0-green.svg)
![License](https://img.shields.io/badge/license-MIT-yellow.svg)

## 📋 概要

このプロジェクトは、ローカル環境で動作するLLM（Large Language Model）と対話するためのWebインターフェースを提供します。Ollamaをバックエンドとして使用し、Streamlitで構築された直感的なUIを通じて、様々なAIモデルと会話できます。

### 🎯 主な特徴

- **🚀 高速レスポンス**: RTX A5000最適化、ストリーミング応答対応
- **🎨 複数モデル対応**: Qwen2.5、Gemma3、GPT-OSS等、切り替え可能
- **💾 会話管理**: 履歴保存、エクスポート、検索機能
- **⚙️ カスタマイズ**: パラメータ調整、キャラクター設定
- **📊 分析機能**: トークン数、応答時間、会話パターン分析
- **🔧 拡張可能**: モジュール設計、プラグイン対応

## 🏗️ プロジェクト構造

```
ollama-streamlit-chat/
├── app/                    # メインアプリケーション
│   ├── simple_chat.py     # 基本チャット機能
│   └── pages/             # マルチページ機能
├── config/                # 設定ファイル
│   ├── models.yaml        # モデル設定
│   └── characters/        # キャラクター定義
├── utils/                 # ユーティリティ
├── data/                  # データ保存
└── scripts/               # 便利スクリプト
```

## 🚀 クイックスタート

### 前提条件

- Ubuntu 24.04 (推奨) / Windows 11 / macOS
- Python 3.11以上
- Conda (Anaconda/Miniconda)
- Ollama v0.3.0以上
- GPU: NVIDIA RTX A5000 (推奨) / 8GB以上のVRAM

### 1. リポジトリのクローン

```bash
git clone https://github.com/yourusername/ollama-streamlit-chat.git
cd ollama-streamlit-chat
```

### 2. 自動セットアップ（推奨）

```bash
chmod +x setup_and_run.sh
./setup_and_run.sh
```

### 3. 手動セットアップ

#### Conda環境の作成

```bash
# 環境作成
conda env create -f environment.yml
conda activate ollama-chat

# または手動インストール
conda create -n ollama-chat python=3.11
conda activate ollama-chat
pip install -r requirements.txt
```

#### Ollamaのセットアップ

```bash
# Ollamaサービス起動
sudo systemctl start ollama

# モデルのダウンロード
ollama pull qwen2.5:7b
ollama pull gemma3:4b
```

#### アプリケーション起動

```bash
# シンプル版
streamlit run app/simple_chat.py

# マルチページ版（フル機能）
streamlit run app/advanced_chat.py
```

## 💻 使い方

### 基本的な使い方

1. ブラウザで `http://localhost:8501` にアクセス
2. サイドバーでモデルとパラメータを選択
3. チャット欄にメッセージを入力して送信
4. ストリーミングで応答が表示されます

### 高度な機能

#### キャラクター設定

```python
# config/characters/custom.json
{
  "name": "アシスタント",
  "personality": "丁寧で親切",
  "system_prompt": "あなたは親切なアシスタントです。",
  "temperature": 0.7,
  "top_p": 0.9
}
```

#### モデル設定

```yaml
# config/models.yaml
models:
  qwen2.5:
    version: "7b"
    context_size: 32768
    gpu_layers: 35
    temperature_default: 0.7
    
  gemma3:
    version: "4b"
    context_size: 8192
    gpu_layers: 28
    temperature_default: 0.8
```

## 📊 パフォーマンス

### RTX A5000での実測値

| モデル | サイズ | VRAM使用量 | トークン/秒 | 初回応答時間 |
|--------|--------|------------|-------------|--------------|
| Qwen2.5 | 7B | 5.2GB | 65-80 | 0.8秒 |
| Gemma3 | 4B | 2.6GB | 90-120 | 0.4秒 |
| Gemma3 | 12B | 7.2GB | 35-50 | 1.2秒 |
| GPT-OSS | 20B | 13.8GB | 25-35 | 2.1秒 |

## 🔧 環境変数

```bash
# .env ファイル
OLLAMA_HOST=localhost:11434
OLLAMA_NUM_PARALLEL=2
OLLAMA_MAX_LOADED_MODELS=3
STREAMLIT_SERVER_PORT=8501
STREAMLIT_THEME=dark
```

## 📦 主な依存関係

- **streamlit**: 1.32.0 - WebUIフレームワーク
- **ollama**: 0.3.0 - Ollama Python クライアント
- **pandas**: データ処理
- **plotly**: グラフ表示
- **pyyaml**: 設定ファイル管理

## 🛠️ トラブルシューティング

### Ollama接続エラー

```bash
# サービス状態確認
sudo systemctl status ollama

# 再起動
sudo systemctl restart ollama

# ログ確認
journalctl -u ollama -f
```

### GPU認識問題

```bash
# NVIDIA-SMI確認
nvidia-smi

# CUDA確認
nvcc --version

# Ollama GPU設定
export OLLAMA_GPU_LAYERS=35
```

### メモリ不足

```bash
# スワップ追加（必要に応じて）
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## 🚧 今後の開発予定

- [ ] 音声入出力対応（VOICEVOX連携）
- [ ] マルチユーザー対応
- [ ] RAG（Retrieval Augmented Generation）実装
- [ ] ファインチューニング機能
- [ ] AI-VTuber連携機能
- [ ] プロンプトエンジニアリングツール
- [ ] モバイル対応UI

## 🤝 コントリビューション

プルリクエストを歓迎します！大きな変更の場合は、まずissueを開いて変更内容を議論してください。

1. フォーク
2. フィーチャーブランチ作成 (`git checkout -b feature/AmazingFeature`)
3. コミット (`git commit -m 'Add some AmazingFeature'`)
4. プッシュ (`git push origin feature/AmazingFeature`)
5. プルリクエスト作成

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 🙏 謝辞

- [Ollama](https://ollama.ai/) - ローカルLLM実行環境
- [Streamlit](https://streamlit.io/) - PythonのWebアプリフレームワーク
- [Qwen2.5](https://github.com/QwenLM/Qwen2.5) - 高性能言語モデル

## 📧 連絡先

- Issue: [GitHub Issues](https://github.com/yourusername/ollama-streamlit-chat/issues)
- Discussion: [GitHub Discussions](https://github.com/yourusername/ollama-streamlit-chat/discussions)

---

**Made with ❤️ for Local AI Community**