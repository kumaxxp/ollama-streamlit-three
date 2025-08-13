# 📚 対応モデル一覧

## 推奨モデル

### 🥇 Qwen2.5-7B
- **用途**: 汎用チャット、コーディング支援、論理的思考
- **VRAM**: 5.2GB
- **速度**: 65-80 tokens/s (RTX A5000)
- **コンテキスト**: 32,768トークン
- **インストール**: `ollama pull qwen2.5:7b`
- **特徴**: 
  - 日本語対応良好
  - コード生成に強い
  - 長文コンテキスト対応

### 🥈 Gemma3-4B
- **用途**: 高速応答、軽量タスク、リアルタイム対話
- **VRAM**: 2.6GB
- **速度**: 90-120 tokens/s (RTX A5000)
- **コンテキスト**: 8,192トークン
- **インストール**: `ollama pull gemma3:4b`
- **特徴**:
  - 超高速レスポンス
  - 低リソース消費
  - キャラクター対話向き

### 🥉 Gemma3-12B
- **用途**: 高品質応答、複雑なタスク
- **VRAM**: 7.2GB
- **速度**: 35-50 tokens/s (RTX A5000)
- **コンテキスト**: 16,384トークン
- **インストール**: `ollama pull gemma3:12b`
- **特徴**:
  - バランス型
  - 品質重視
  - 創造的タスク対応

## その他の対応モデル

### GPT-OSS-20B
- **VRAM**: 13.8GB
- **速度**: 25-35 tokens/s
- **インストール**: `ollama pull gpt-oss:20b`
- **特徴**: 高品質、大規模モデル

### Llama3.1-8B
- **VRAM**: 5.5GB
- **速度**: 60-75 tokens/s
- **インストール**: `ollama pull llama3.1:8b`
- **特徴**: Meta社製、多言語対応

### Mistral-7B
- **VRAM**: 4.5GB
- **速度**: 70-85 tokens/s
- **インストール**: `ollama pull mistral:7b`
- **特徴**: 欧州製、高速推論

## モデル選択ガイド

### 用途別推奨

| 用途 | 推奨モデル | 理由 |
|------|------------|------|
| 日常会話 | Qwen2.5-7B | バランスが良い |
| リアルタイムチャット | Gemma3-4B | 超高速応答 |
| コード生成 | Qwen2.5-7B | プログラミング知識豊富 |
| 創作活動 | Gemma3-12B | 創造性が高い |
| 長文処理 | Qwen2.5-7B | 32Kコンテキスト |
| 省リソース | Gemma3-4B | 低VRAM使用 |

### VRAM別推奨構成

#### 8GB VRAM
- 単一モデル: Qwen2.5-7B または Gemma3-12B
- 複数モデル: Gemma3-4B × 2

#### 16GB VRAM
- 単一モデル: GPT-OSS-20B
- 複数モデル: Qwen2.5-7B + Gemma3-4B

#### 24GB VRAM (RTX A5000)
- 単一モデル: 任意の組み合わせ
- 複数モデル: 最大3モデル同時ロード可能

## パラメータ調整ガイド

### Temperature（創造性）
- `0.0-0.3`: 事実ベース、確定的
- `0.4-0.7`: バランス型（推奨）
- `0.8-1.0`: 創造的、変化に富む
- `1.1-2.0`: 実験的、予測不可能

### Top P（多様性）
- `0.1-0.4`: 限定的な語彙
- `0.5-0.9`: 標準的（推奨）
- `0.95-1.0`: 最大の多様性

### Repeat Penalty（繰り返し抑制）
- `1.0`: 無効（Gemma3推奨）
- `1.05-1.15`: 軽度の抑制
- `1.2-1.5`: 強い抑制

## モデル管理コマンド

```bash
# モデル一覧表示
ollama list

# モデルダウンロード
ollama pull <model:tag>

# モデル削除
ollama rm <model:tag>

# モデル情報表示
ollama show <model:tag>

# カスタムモデル作成
ollama create mymodel -f Modelfile
```

## カスタムモデルの作成

### Modelfile例

```dockerfile
# ベースモデル指定
FROM qwen2.5:7b

# パラメータ設定
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096

# システムプロンプト
SYSTEM """
あなたは親切で知識豊富なアシスタントです。
ユーザーの質問に対して、正確で分かりやすい回答を提供してください。
"""

# テンプレート設定
TEMPLATE """{{ .System }}
User: {{ .Prompt }}
Assistant: {{ .Response }}"""
```

### カスタムモデル作成

```bash
# Modelfile作成
cat > Modelfile << EOF
FROM qwen2.5:7b
PARAMETER temperature 0.7
SYSTEM "カスタムアシスタント"
EOF

# モデル作成
ollama create my-assistant -f Modelfile

# 使用
ollama run my-assistant
```

## トラブルシューティング

### モデルが遅い
1. GPU使用確認: `nvidia-smi`
2. レイヤー設定: `OLLAMA_GPU_LAYERS=35`
3. 小さいモデルに変更

### メモリ不足
1. 小さいモデル使用
2. コンテキストサイズ削減
3. 並列実行数制限

### 品質が低い
1. Temperature調整
2. より大きいモデル使用
3. プロンプト改善

## 関連リンク

- [Ollama公式](https://ollama.ai/)
- [モデルライブラリ](https://ollama.ai/library)
- [Hugging Face](https://huggingface.co/)
- [LLMベンチマーク](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)