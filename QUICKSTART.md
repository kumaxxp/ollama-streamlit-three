# 🚀 Advanced Dialogue System クイックスタート

## 📋 必要な環境
- Python 3.11+
- Ollama インストール済み
- GPU VRAM 6GB以上（推奨）

## 🎯 推奨モデルセットアップ

### まずは自動インストール（推奨）
config/model_config.json に基づいて推奨モデルを一括インストールできます。

```bash
# 計画の確認
python scripts/install_models.py --list --include-defaults

# 推奨モデルの取得（既存はスキップ）
python scripts/install_models.py --pull --include-defaults --skip-available
```

手動で進めたい場合は、以下のステップに従ってください。

### ステップ1: Ollamaの確認
```bash
# Ollamaが起動しているか確認
ollama list

# 起動していない場合
ollama serve
```

### ステップ2: 推奨モデルのインストール

#### 最小構成（VRAM 6GB）
```bash
# エージェント用（必須）
ollama pull qwen2.5:7b-instruct-q4_K_M

# Director用（必須）
ollama pull gemma3:4b
```

#### 標準構成（VRAM 10GB）
```bash
# 最小構成に加えて
ollama pull gemma3:12b
```

#### フル構成（VRAM 16GB以上）
```bash
# 全推奨モデル
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull gemma3:4b
ollama pull gemma3:12b
ollama pull gpt-oss:20b
ollama pull qwen:7b
```

### ステップ3: モデル確認
```bash
# インストール済みモデルの確認
python check_models.py
```

### ステップ4: アプリケーション起動
```bash
# Streamlitアプリを起動
streamlit run app/pages/03_Advanced_Dialogue_Refactored.py
```

## 🎭 モデル選択ガイド

### エージェント用モデル

| 状況 | 推奨モデル | 理由 |
|------|-----------|------|
| **通常の日本語対話** | `qwen2.5:7b-instruct-q4_K_M` | 日本語性能◎、速度◎、メモリ効率◎ |
| **複雑な議論** | `gemma3:12b` | 推論能力が高い |
| **創造的な対話** | `gpt-oss:20b` | 創造性が高い（要16GB VRAM） |

### Director用モデル

| 状況 | 推奨モデル | 理由 |
|------|-----------|------|
| **通常使用** | `gemma3:4b` | 高速判断、低レイテンシ |
| **詳細分析が必要** | `qwen2.5:7b-instruct-q4_K_M` | より深い分析が可能 |

## ⚙️ 推奨設定

### 温度設定
- **エージェント**: 0.7（バランス型）
- **Director**: 0.3（一貫性重視）

### ターン数
- **短い議論**: 10-15ターン
- **標準的な議論**: 20ターン
- **深い議論**: 25-30ターン

## 🔧 トラブルシューティング

### モデルが見つからない
```bash
# 具体的なモデル名でインストール
ollama pull qwen2.5:7b-instruct-q4_K_M

# タグを確認
ollama list
```

### メモリ不足エラー
- より小さいモデルを使用（`gemma3:4b`など）
- 量子化レベルの高いモデルを選択（q4_K_M推奨）

### 応答が遅い
- Director温度を0.2に下げる
- `gemma3:4b`をDirectorに使用
- ターン間の待機時間を調整

## 📊 パフォーマンス比較

| モデル | 応答速度 | 品質 | VRAM使用 |
|--------|----------|------|----------|
| qwen2.5:7b-instruct-q4_K_M | 高速 | 高 | 4.7GB |
| gemma3:12b | 中速 | 高 | 8.1GB |
| gpt-oss:20b | 低速 | 最高 | 13GB |
| gemma3:4b | 最高速 | 中 | 3.3GB |

## 📝 使用例

### 基本的な対話
1. エージェント: `qwen2.5:7b-instruct-q4_K_M`
2. Director: `gemma3:4b`
3. 温度: エージェント0.7、Director0.3
4. キャラクター: お好みで選択

### 高品質な議論
1. エージェント: `gemma3:12b`
2. Director: `qwen2.5:7b-instruct-q4_K_M`
3. 温度: エージェント0.6、Director0.3
4. ターン数: 25以上

## 🆘 サポート
問題が発生した場合は、GitHubのIssueにご報告ください。
https://github.com/kumaxxp/ollama-streamlit-three