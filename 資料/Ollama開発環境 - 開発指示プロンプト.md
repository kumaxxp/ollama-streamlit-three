# 🤖 Ollama開発環境 - 開発指示プロンプト

## 📋 プロジェクト概要

私はOllama + Streamlitベースのローカルチャットシステムを開発しています。以下のGitHubリポジトリをベースに、新機能の追加や改善を行ってください。

**リポジトリ**: https://github.com/kumaxxp/ollama-streamlit-chat

## 🖥️ 開発環境

```yaml
OS: Ubuntu 24.04 LTS
GPU: NVIDIA RTX A5000 (24GB VRAM)
Python: 3.11
Conda環境: ollama-chat
Ollamaバージョン: 0.3.0以上
主要フレームワーク:
  - Streamlit 1.32.0
  - Ollama Python Client 0.3.0
```

## 🚀 環境セットアップ

```bash
# 1. リポジトリをクローン
git clone https://github.com/kumaxxp/ollama-streamlit-chat.git
cd ollama-streamlit-chat

# 2. Conda環境を有効化（既存環境を使用）
conda activate ollama-chat

# 3. Ollamaサービス確認
sudo systemctl status ollama

# 4. 開発サーバー起動
streamlit run app/simple_chat.py
```

## 📂 プロジェクト構造

```
ollama-streamlit-chat/
├── app/                 # Streamlitアプリ本体
│   ├── simple_chat.py   # 基本チャット機能（実装済み）
│   └── pages/           # マルチページ機能（拡張用）
├── config/              # 設定ファイル
├── utils/               # ユーティリティ
├── data/                # データ保存
└── scripts/             # 便利スクリプト
```

## ✅ 実装済み機能

1. **基本チャット機能** (`simple_chat.py`)
   - Ollamaモデルとのリアルタイム対話
   - ストリーミング応答
   - パラメータ調整（Temperature, Top P等）
   - 会話履歴管理
   - JSON形式での履歴保存

2. **対応モデル**
   - Qwen2.5-7B（推奨）
   - Gemma3-4B/12B
   - その他Ollamaモデル

## 🔧 開発ガイドライン

### 新機能追加時の注意事項

1. **Conda環境の共有使用**
   ```bash
   # 必ず既存のollama-chat環境を使用
   conda activate ollama-chat
   
   # 新しいパッケージ追加時
   pip install <package>
   pip freeze > requirements.txt  # 更新を忘れずに
   ```

2. **Ollama API使用**
   ```python
   import ollama
   
   # 基本的な使用方法
   response = ollama.chat(
       model="qwen2.5:7b",
       messages=[{"role": "user", "content": "Hello"}],
       stream=True  # ストリーミング推奨
   )
   ```

3. **Streamlitベストプラクティス**
   ```python
   # セッション状態を活用
   if "messages" not in st.session_state:
       st.session_state.messages = []
   
   # キャッシュを適切に使用
   @st.cache_resource
   def load_model():
       return ollama.list()
   ```

4. **GPU最適化**
   - RTX A5000の24GB VRAMを考慮
   - 複数モデルの同時ロード可能（最大3モデル）
   - Flash Attention有効

## 📋 開発タスク例

### 機能拡張の優先順位

1. **高優先度**
   - [ ] キャラクター設定機能（config/characters/）
   - [ ] 会話分析ページ（pages/Analytics.py）
   - [ ] プロンプトテンプレート管理

2. **中優先度**
   - [ ] RAG（Retrieval Augmented Generation）実装
   - [ ] マルチモーダル対応（画像入力）
   - [ ] 会話のDB保存（SQLite）

3. **低優先度**
   - [ ] 音声入出力（VOICEVOX連携）
   - [ ] AI-VTuber機能
   - [ ] モバイルUI最適化

## ⚠️ 制約事項

1. **Ollamaサービス依存**
   - 開発前に必ずOllamaが起動していることを確認
   - ポート11434がデフォルト

2. **モデル特性**
   - Gemma3: `repeat_penalty`は1.0固定
   - Qwen2.5: 日本語性能が高い
   - コンテキスト長に注意（モデルにより異なる）

3. **パフォーマンス考慮**
   - ストリーミング応答を基本とする
   - 大きなモデルは初回ロード時間が長い

## 🛠️ 開発コマンド集

```bash
# Ollama関連
ollama list                      # モデル一覧
ollama pull <model>              # モデルダウンロード
sudo systemctl restart ollama    # サービス再起動
journalctl -u ollama -f          # ログ監視

# 開発関連
streamlit run app/simple_chat.py --server.port 8501
python -m pytest tests/          # テスト実行
black app/                       # コード整形

# GPU監視
watch -n 1 nvidia-smi           # GPU使用状況
```

## 📝 コミット規約

```
feat: 新機能追加
fix: バグ修正
docs: ドキュメント更新
style: コード整形
refactor: リファクタリング
test: テスト追加
chore: その他の変更
```

## 🔗 関連情報

- **過去の検討内容**：
  - Gemma3 Character Builder実装経験
  - 3エージェント対話システム
  - プロンプトエンジニアリング最適化

- **技術スタック**：
  - Python実装中心
  - ローカルLLM（Ollama経由）
  - WebUI（Streamlit）
  - GPU最適化（RTX A5000）

## 💡 開発のヒント

1. まず`simple_chat.py`のコードを理解してから拡張
2. 新機能は`pages/`ディレクトリに追加を推奨
3. 設定は`config/`に外部化する
4. ユーティリティは`utils/`にモジュール化
5. テストを書く習慣をつける

---

**このプロンプトを新しいClaude会話の最初に貼り付けることで、開発環境と制約を理解した上で適切な支援が受けられます。**