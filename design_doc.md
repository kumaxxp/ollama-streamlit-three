# Advanced Dialogue System 設計書

## 1. システム概要

### 1.1 目的
2つのLLMエージェントが議論を行い、Director（監督AI）が議論の質を管理・向上させる対話生成システム。プログラム的な制御から、LLMの能力を最大限活用したプロンプトエンジニアリングベースの制御へ移行。

### 1.2 主要コンポーネント
- **Director (監督AI)**: 議論の分析、戦略選択、介入指示
- **Agent (対話エージェント)**: キャラクター設定に基づく議論参加者
- **Dialogue Manager**: 対話フロー全体の管理

### 1.3 特徴
- **クオリティ重視**: 毎ターンDirectorが介入し議論品質を向上
- **キャラクター駆動**: 外部JSON設定による柔軟なペルソナ管理
- **戦略的介入**: 5つの介入戦略と4つの議論フェーズ
- **モジュール設計**: UIとロジックの完全分離

## 2. アーキテクチャ

### 2.1 システム構成図
```
┌─────────────────────────────────────┐
│         03_Advanced_Dialogue.py      │
│              (UI Layer)              │
└─────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────┐
│         DialogueManager              │
│    (Orchestration Layer)             │
└─────────────────────────────────────┘
        ↙                    ↘
┌──────────────┐      ┌──────────────┐
│   Director   │      │    Agents    │
│ (Control AI) │      │ (Dialog AI)  │
└──────────────┘      └──────────────┘
        ↓                      ↓
┌─────────────────────────────────────┐
│          Configuration Files         │
│   (characters/strategies/prompts)    │
└─────────────────────────────────────┘
```

### 2.2 データフロー
1. **初期化**: テーマ選択 → キャラクター選択 → Director初期化
2. **対話ループ**:
   - Director: 現状分析
   - Director: 戦略選択・指示生成
   - Agent A: 指示を踏まえた発言
   - Agent B: 応答生成
   - Director: 品質評価
3. **終了**: 設定ターン数到達 or Director判断

### 2.3 ファイル構造
```
ollama-streamlit-three/
├── app/
│   ├── pages/
│   │   ├── 01_Simple_Chat.py
│   │   ├── 02_Dynamic_Dialogue.py
│   │   └── 03_Advanced_Dialogue.py   # UI層
│   └── core/
│       ├── agent.py                  # エージェント実装
│       ├── director.py               # 監督AI実装
│       └── dialogue_manager.py       # 統合管理
├── config/
│   ├── characters.json              # キャラクター定義
│   ├── strategies.json              # 議論戦略定義
│   └── prompt_templates.json        # プロンプトテンプレート
└── data/
    └── dialogues/                   # 保存された対話
```

## 3. 主要クラス設計

### 3.1 Agent クラス
**責務**: キャラクター設定に基づく応答生成

**主要メソッド**:
- `__init__(agent_id, character_type, model_name, temperature)`
- `set_session_context(theme, goal, phase)`: セッション設定
- `add_directive(instruction, attention_points)`: Director指示追加
- `generate_response(opponent_message)`: 応答生成
- `build_prompt(opponent_message)`: プロンプト構築

**状態管理**:
- `character`: キャラクター設定
- `session_context`: セッション情報
- `turn_directives`: Director指示履歴
- `memory`: 発言履歴

### 3.2 Director クラス
**責務**: 議論の分析・戦略選択・介入

**主要メソッド**:
- `analyze_dialogue(history)`: 対話状態分析
- `select_strategy(analysis, phase)`: 戦略選択
- `generate_instruction(strategy, agent_id)`: 指示生成
- `evaluate_response(response, instruction)`: 応答評価

**分析指標**:
- `depth_level`: 議論の深さ (1-5)
- `divergence`: 発散度 (0-1)
- `conflict_level`: 対立度 (0-1)
- `productivity`: 生産性 (0-1)

### 3.3 DialogueManager クラス
**責務**: 対話フロー全体の管理

**主要メソッド**:
- `initialize(theme, agent1_type, agent2_type)`: 初期化
- `run_turn()`: 1ターン実行
- `run_dialogue(max_turns)`: 対話全体実行
- `get_phase()`: 現在フェーズ取得
- `save_dialogue()`: 対話保存

## 4. プロンプト設計

### 4.1 3層プロンプト構造
1. **Character Foundation** (不変層)
   - personality: 基本性格
   - expertise: 専門分野
   - behavioral_rules: 行動規則

2. **Session Context** (セッション層)
   - theme: 議論テーマ
   - goal: 議論目標
   - phase: 現在フェーズ

3. **Turn Directive** (ターン層)
   - instruction: Director指示
   - attention_points: 注意事項
   - recent_context: 直近文脈

### 4.2 Director プロンプト
- **分析プロンプト**: 対話状態をJSON形式で分析
- **戦略選択プロンプト**: 最適戦略と具体的指示を生成
- **評価プロンプト**: 応答品質を0-10で評価

## 5. 議論戦略

### 5.1 介入戦略
1. **deepening**: 議論の深化
2. **convergence**: 収束誘導
3. **perspective_shift**: 視点転換
4. **constructive_conflict**: 建設的対立
5. **synthesis**: 統合と発展

### 5.2 議論フェーズ
1. **exploration** (探索): 5ターン
2. **deepening** (深化): 5ターン
3. **convergence** (収束): 5ターン
4. **synthesis** (統合): 5ターン

## 6. キャラクター設定

### 6.1 実装済みキャラクター
- **philosophical_socrates**: 哲学者ソクラテス
- **scientific_darwin**: 科学者ダーウィン
- **creative_artist**: 創造的芸術家
- **pragmatic_engineer**: 実践的エンジニア

### 6.2 キャラクター構造
```json
{
  "name": "表示名",
  "personality": {
    "base": "基本設定",
    "traits": ["特性リスト"],
    "communication_style": "コミュニケーションスタイル"
  },
  "expertise": ["専門分野"],
  "behavioral_rules": ["行動規則"],
  "speaking_patterns": {
    "opening": ["開始パターン"],
    "challenging": ["挑戦パターン"],
    "agreeing": ["同意パターン"]
  }
}
```

## 7. 実装上の工夫

### 7.1 クオリティ重視設計
- 毎ターンDirector介入による品質管理
- 詳細な分析指標による状態把握
- 戦略的介入による議論誘導

### 7.2 拡張性
- JSON外部化による設定管理
- モジュール分離による保守性
- 新キャラクター・戦略の追加容易性

### 7.3 エラーハンドリング
- 各層でのフォールバック実装
- デフォルト値の設定
- ユーザーへの適切なフィードバック

## 8. 使用方法

### 8.1 基本フロー
1. Streamlitアプリケーションを起動
2. 「03_Advanced_Dialogue」ページを選択
3. テーマとキャラクターを選択
4. パラメータを調整（任意）
5. 「対話を開始」ボタンをクリック
6. 生成された対話を確認・保存

### 8.2 カスタマイズ
- `config/characters.json`: 新キャラクター追加
- `config/strategies.json`: 新戦略追加
- `config/prompt_templates.json`: プロンプト調整

## 9. 今後の拡張可能性

### 9.1 Phase 2 候補
- マルチエージェント対応（3人以上）
- 動的キャラクター生成
- 学習機能の追加
- リアルタイム編集機能

### 9.2 Phase 3 候補
- 外部知識ベース連携
- 音声対話対応
- 感情分析統合
- 自動要約機能

## 10. 技術仕様

### 10.1 依存関係
- Python 3.11+
- Streamlit 1.32.0+
- Ollama Python Client 0.3.0+
- 推奨モデル: Qwen2.5-7B, Gemma3-4B

### 10.2 推奨環境
- GPU: 8GB+ VRAM
- RAM: 16GB+
- ストレージ: 10GB+

## 11. ライセンスと貢献

このプロジェクトはMITライセンスの下で公開されています。
貢献を歓迎します。Pull RequestやIssueをGitHubリポジトリまでお寄せください。

---
作成日: 2024年
バージョン: 1.0.0