#!/usr/bin/env python3
"""
長さ制御機能のテストスクリプト
Phase 1実装の動作確認用
"""

import asyncio
import json
from app.core.dialogue_manager import DialogueManager
from app.core.director import AutonomousDirector
import ollama

async def test_length_control():
    """長さ制御機能をテスト"""
    
    print("=" * 60)
    print("📏 長さ制御機能テスト")
    print("=" * 60)
    
    # Ollamaクライアント初期化
    client = ollama.Client()
    
    # DialogueManager初期化
    manager = DialogueManager(client, director_model="gemma3:4b")
    
    # テスト用の対話履歴を作成
    test_dialogues = [
        # 短い応答のパターン
        {
            "dialogues": [
                {"speaker": "やな", "listener": "あゆ", "message": "うん"},
                {"speaker": "あゆ", "listener": "やな姉さま", "message": "はい"},
                {"speaker": "やな", "listener": "あゆ", "message": "そうだね"},
            ],
            "expected": "詳細",
            "scenario": "短すぎる応答"
        },
        # 長い応答のパターン
        {
            "dialogues": [
                {"speaker": "やな", "listener": "あゆ", "message": "私のデータベースによると、" + "これは非常に興味深い問題で、" * 20},
                {"speaker": "あゆ", "listener": "やな姉さま", "message": "確かにその通りですが、" + "別の観点から見ると、" * 20},
            ],
            "expected": "簡潔",
            "scenario": "長すぎる応答"
        },
        # バランスの悪いパターン
        {
            "dialogues": [
                {"speaker": "やな", "listener": "あゆ", "message": "これについて私の意見を詳しく説明すると" + "、" * 50},
                {"speaker": "あゆ", "listener": "やな姉さま", "message": "なるほど"},
                {"speaker": "やな", "listener": "あゆ", "message": "さらに付け加えると" + "、" * 40},
                {"speaker": "あゆ", "listener": "やな姉さま", "message": "確かに"},
            ],
            "expected": "imbalanced",
            "scenario": "アンバランスな対話"
        }
    ]
    
    for i, test_case in enumerate(test_dialogues, 1):
        print(f"\n--- テストケース {i}: {test_case['scenario']} ---")
        
        # 長さ分析を実行
        analysis = manager.director.analyze_response_lengths(test_case['dialogues'])
        
        print(f"平均長さ: {analysis['average_length']:.0f}文字")
        print(f"トレンド: {analysis['recent_trend']}")
        print(f"バランス: {analysis['balance']}")
        print(f"推奨: {analysis['recommendation']}")
        
        # Director評価を実行
        evaluation = await manager.director.evaluate_dialogue(test_case['dialogues'])
        
        print(f"\nDirector判断:")
        print(f"  介入必要: {evaluation.get('intervention_needed', False)}")
        print(f"  理由: {evaluation.get('reason', '不明')}")
        print(f"  タイプ: {evaluation.get('intervention_type', 'なし')}")
        print(f"  長さガイド: {evaluation.get('response_length_guide', '現状維持')}")
        
        if evaluation.get('message'):
            print(f"  メッセージ: {evaluation['message']}")
        
        # 長さ指示の生成テスト
        if evaluation.get('response_length_guide') != '現状維持':
            instruction = manager.director.generate_length_instruction(
                evaluation['response_length_guide']
            )
            print(f"\n生成された長さ指示:")
            print(instruction)
    
    print("\n" + "=" * 60)
    print("✅ テスト完了")

def test_prompt_generation():
    """長さ指示を含むプロンプト生成のテスト"""
    
    print("\n" + "=" * 60)
    print("📝 プロンプト生成テスト")
    print("=" * 60)
    
    from app.core.agent import Agent
    
    # テスト用エージェント作成
    agent = Agent(
        agent_id="test",
        character_type="AI-tuber-college_student_girl",
        model_name="qwen2.5:7b-instruct-q4_K_M",
        ollama_client=ollama.Client()
    )
    
    # 長さ指示を追加
    test_cases = [
        ("簡潔", ["50-100文字で簡潔に", "要点のみ"]),
        ("標準", ["100-200文字程度で", "適度な詳しさ"]),
        ("詳細", ["200-300文字で詳しく", "具体例を含めて"])
    ]
    
    for length_type, attention_points in test_cases:
        print(f"\n--- {length_type}の場合 ---")
        
        agent.add_directive(
            f"次の応答は{length_type}にまとめてください",
            attention_points
        )
        
        context = {
            "opponent_name": "あゆ",
            "opponent_message": "やな姉さま、AIの感情について教えてください",
            "recent_history": [],
            "director_instruction": ""
        }
        
        prompt = agent.build_prompt(context)
        
        # 長さ指示部分を抽出して表示
        if "【応答の長さ】" in prompt:
            length_part = prompt.split("【応答の長さ】")[1].split("\n")[0]
            print(f"長さ指示: {length_part}")
        else:
            print("長さ指示が含まれていません")
    
    print("\n" + "=" * 60)
    print("✅ プロンプト生成テスト完了")

if __name__ == "__main__":
    # 非同期テストを実行
    print("\n🚀 長さ制御機能のテストを開始します\n")
    
    try:
        # 長さ分析とDirector評価のテスト
        asyncio.run(test_length_control())
        
        # プロンプト生成のテスト
        test_prompt_generation()
        
        print("\n✨ 全テスト完了！")
        print("\n次のステップ:")
        print("1. streamlit run app/pages/03_Advanced_Dialogue.py")
        print("2. 応答長制御を「簡潔」「標準」「詳細」で試す")
        print("3. 自動長さバランス調整をON/OFFで比較")
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()