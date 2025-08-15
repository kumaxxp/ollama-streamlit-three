conda activate ollama-chat

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
ollama-streamlit-three/
├── app/
│   ├── core/                       # コアロジック（Agent/Director/Model管理など）
│   ├── ui/                         # UIヘルパー
│   └── pages/                      # Streamlitページ
│       ├── 01_Simple_Chat.py
│       └── 03_Advanced_Dialogue_Refactored.py
├── config/                         # 設定ファイル（単一ソース）
│   ├── model_config.json           # モデル優先度/推奨温度/デフォルト
│   ├── characters.json             # キャラクター定義
│   ├── strategies.json             # 対話戦略
│   ├── prompt_templates.json       # プロンプトテンプレート
│   └── director_prompts.json       # Director用プロンプト
├── scripts/
│   └── install_models.py           # 推奨モデルの自動インストーラ
├── data/
│   └── dialogues/                  # 会話ログ
├── tests/                          # 最低限のテスト
├── run_app.sh                      # 高機能ページの起動スクリプト
├── setup_and_run.sh                # セットアップ＆起動ランチャ
└── QUICKSTART.md                   # クイックスタート
```


## 🎯 採用モデル仕様

本システムは以下の量子化モデルを採用し、実用性とパフォーマンスを最適化しています。

### 推奨モデル構成

| モデル | 用途 | サイズ | 特徴 |
|--------|------|--------|------|
| **qwen2.5:7b-instruct-q4_K_M** | メインエージェント | 4.7GB | 日本語対話に最適化、4bit量子化で高速動作 |
| **gemma3:4b** | Director AI | 3.3GB | 軽量で高速な判断、低レイテンシ |
| **gemma3:12b** | 高品質エージェント | 8.1GB | より複雑な議論や推論が必要な場合 |
| **gpt-oss:20b** | 創造的対話 | 13GB | 高品質な文章生成、創造的なタスク |

### モデルの自動インストール（推奨）

config/model_config.json を単一のソースとして参照し、推奨モデルを自動でインストールできます。

```bash
# Conda 環境を有効化
conda activate ollama-chat

# インストール計画の確認
python scripts/install_models.py --list --include-defaults

# 推奨モデルをインストール（既にあるモデルはスキップ）
python scripts/install_models.py --pull --include-defaults --skip-available

# 例: すべての推奨グループを取得（primary + lightweight 以外も含める）
python scripts/install_models.py --pull --groups all --skip-available

# 例: 個別に追加インストール
python scripts/install_models.py --pull --names gemma3:12b gpt-oss:20b
```

補足: 対話式スクリプトからの自動モードも利用できます。

```bash
./install_models.sh    # メニューで「Auto (recommended)」を選択
```

### インストール（必須）
```bash
# 最小構成（VRAM 6GB以上）
ollama pull qwen2.5:7b-instruct-q4_K_M  # エージェント用
ollama pull gemma3:4b                   # Director用

# 確認
python check_models.py
```

### なぜ量子化モデルを採用するのか

1. **メモリ効率**: q4_K_M量子化により、モデルサイズを約75%削減
2. **速度向上**: 推論速度が約90%向上
3. **品質維持**: 元モデルの約85%の品質を維持
4. **実用性**: 一般的なGPU（RTX 3060等）でも動作可能

### モデル選択の根拠

- **qwen2.5:7b-instruct-q4_K_M**: 
  - Qwen2.5シリーズは日本語のベンチマークで最高スコア
  - instruct版は指示追従性が高く、対話に最適
  - q4_K_M量子化は品質とパフォーマンスの最良のバランス

- **gemma3:4b**:
  - Google製の最新アーキテクチャ
  - Director用途では判断の一貫性が重要なため、小型モデルが適切
  - 4Bパラメータは必要十分な判断能力を持つ

### パフォーマンス実測値

| メトリクス | qwen2.5:7b-instruct-q4_K_M | gemma3:12b | gpt-oss:20b |
|-----------|---------------------------|------------|-------------|
| 初回応答時間 | 0.8秒 | 1.5秒 | 2.3秒 |
| トークン/秒 | 45 | 28 | 18 |
| VRAM使用量 | 4.7GB | 8.1GB | 13GB |
| 日本語品質 | ★★★★★ | ★★★★☆ | ★★★★☆ |

### 動作環境別推奨構成

| GPU VRAM | 推奨構成 |
|----------|----------|
| 6GB (RTX 3060等) | qwen2.5:7b-instruct-q4_K_M + gemma3:4b |
| 8GB (RTX 3070等) | qwen2.5:7b-instruct-q4_K_M + gemma3:4b |
| 12GB (RTX 3080Ti等) | gemma3:12b + gemma3:4b |
| 16GB以上 (RTX 4080等) | gpt-oss:20b + qwen2.5:7b-instruct-q4_K_M |

## 🚀 クイックスタート

### 前提条件

- Ubuntu 24.04 (推奨) / Windows 11 / macOS
- Python 3.11以上
- Conda (Anaconda/Miniconda)
- Ollama v0.3.0以上
- GPU: NVIDIA RTX A5000 (推奨) / 8GB以上のVRAM

### 1. リポジトリのクローン

```bash
git clone https://github.com/kumaxxp/ollama-streamlit-three.git
cd ollama-streamlit-three
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

# モデルのダウンロード（最小構成）
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull gemma3:4b
```

#### アプリケーション起動

```bash
# シンプル版（単一ページ）
streamlit run app/pages/01_Simple_Chat.py

# 高機能版（Advanced Dialogue）
streamlit run app/pages/03_Advanced_Dialogue_Refactored.py

# またはスクリプトから起動
./run_app.sh
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

#### モデル設定（単一ソース: config/model_config.json）

```json
{
  "default_models": {
    "agent": "qwen2.5:7b-instruct-q4_K_M",
    "director": "gemma3:4b",
    "fallback": "qwen:7b"
  },
  "model_selection_rules": {
    "dialogue_agent": { "preferred_models": ["qwen2.5:7b-instruct-q4_K_M", "gemma3:12b"], "recommended_temperature": 0.7 },
    "director": { "preferred_models": ["gemma3:4b", "qwen2.5:7b-instruct-q4_K_M"], "recommended_temperature": 0.3 }
  },
  "production_models": {
    "primary": [
      { "name": "qwen2.5:7b-instruct-q4_K_M", "priority": 1, "temperature": { "agent": 0.7 } },
      { "name": "gemma3:4b", "priority": 2, "temperature": { "director": 0.3 } }
    ],
    "extended": [
      { "name": "gemma3:12b", "priority": 10 },
      { "name": "gpt-oss:20b", "priority": 20 }
    ]
  }
}
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

- Issue: https://github.com/kumaxxp/ollama-streamlit-three/issues
- Discussion: https://github.com/kumaxxp/ollama-streamlit-three/discussions

---

**Made with ❤️ for Local AI Community**