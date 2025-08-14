#!/usr/bin/env python3
"""
会話スタイル改善の確認テスト
挨拶とマークダウン形式の抑制を確認
"""

import asyncio
import json
import re
from app.core.dialogue_manager import DialogueManager
from app.core.agent import Agent
import ollama

def check_greeting_patterns(text: str) -> bool:
    """挨拶パターンをチェック"""
    greeting_patterns = [
        r'こんにちは',
        r'よろしく',
        r'はじめまして',
        r'お疲れ様',
        r'おはよう',
        r'こんばんは',
        r'初めまして'
    ]
    
    for pattern in greeting_patterns:
        if re.search(pattern, text):
            return True
    return False

def check_markdown_patterns(text: str) -> bool:
    """マークダウン記号をチェック"""
    markdown_patterns = [
        r'^\s*[-*・]\s+',  # 箇条書き記号
        r'#{1,6}\s+',      # 見出し
        r'\*\*.*?\*\*',    # 太字
        r'^\d+\.\s+',       # 番号付きリスト
        r'^\s*>\s+',        # 引用
        r'```',             # コードブロック
    ]
    
    for pattern in markdown_patterns:
        if re.search(pattern, text, re.MULTILINE):
            return True
    return False

async def test_conversation_style():
    """会話スタイルのテスト"""
    
    print("=" * 60)
    print("💬 会話スタイル改善テスト")
    print("=" * 60)
    
    # Ollamaクライアント初期化
    client = ollama.Client()
    
    # AI姉妹でテスト
    print("\n--- AI姉妹の対話テスト ---")
    
    manager = DialogueManager(client, director_model="gemma3:4b")
    
    # AI姉妹の設定
    agent1_config = {
        'character_type': 'AI-tuber-college_student_girl',  # やな
        'model': 'qwen2.5:7b-instruct-q4_K_M',
        'temperature': 0.7
    }
    agent2_config = {
        'character_type': 'AI-tuber-high_school_girl',  # あゆ
        'model': 'qwen2.5:7b-instruct-q4_K_M',
        'temperature': 0.5
    }
    
    # 初期化
    manager.initialize(
        "AIの感情について",
        agent1_config,
        agent2_config
    )
    
    print(f"テーマ: AIの感情について")
    print(f"参加者: {manager.agent1.character['name']} & {manager.agent2.character['name']}")
    print("-" * 40)
    
    # 5ターン実行してチェック
    greeting_count = 0
    markdown_count = 0
    
    for turn in range(5):
        print(f"\n【ターン {turn + 1}】")
        
        result = await manager.run_turn()
        
        speaker = result.get('speaker', '不明')
        message = result.get('message', '')
        
        print(f"{speaker}: {message[:100]}..." if len(message) > 100 else f"{speaker}: {message}")
        
        # チェック
        if turn > 0:  # 最初のターンは挨拶OKとする
            if check_greeting_patterns(message):
                greeting_count += 1
                print("  ⚠️ 挨拶が検出されました")
        
        if check_markdown_patterns(message):
            markdown_count += 1
            print("  ⚠️ マークダウン形式が検出されました")
            # どのパターンか詳細表示
            if re.search(r'^\s*[-*・]\s+', message, re.MULTILINE):
                print("    - 箇条書き記号")
            if re.search(r'#{1,6}\s+', message):
                print("    - 見出し記号")
            if re.search(r'\*\*.*?\*\*', message):
                print("    - 太字記号")
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("📊 テスト結果")
    print("-" * 40)
    print(f"✅ 総ターン数: 5")
    print(f"{'❌' if greeting_count > 0 else '✅'} 不要な挨拶: {greeting_count}回")
    print(f"{'❌' if markdown_count > 0 else '✅'} マークダウン形式: {markdown_count}回")
    
    if greeting_count == 0 and markdown_count == 0:
        print("\n🎉 完璧！自然な会話スタイルが実現されています")
    else:
        print("\n💡 改善の余地があります")
        if greeting_count > 0:
            print("  - 挨拶の抑制をさらに強化する必要があります")
        if markdown_count > 0:
            print("  - マークダウン記号の使用を避ける指示を強化する必要があります")

async def test_general_characters():
    """一般キャラクターの会話スタイルテスト"""
    
    print("\n\n" + "=" * 60)
    print("👥 一般キャラクターの会話スタイルテスト")
    print("=" * 60)
    
    client = ollama.Client()
    manager = DialogueManager(client)
    
    # 高校生と会社員でテスト
    agent1_config = {
        'character_type': 'high_school_girl_optimistic',
        'model': 'qwen2.5:7b-instruct-q4_K_M',
        'temperature': 0.8
    }
    agent2_config = {
        'character_type': 'office_worker_tired',
        'model': 'qwen2.5:7b-instruct-q4_K_M',
        'temperature': 0.6
    }
    
    manager.initialize(
        "幸せな人生とは",
        agent1_config,
        agent2_config
    )
    
    print(f"テーマ: 幸せな人生とは")
    print(f"参加者: {manager.agent1.character['name']} & {manager.agent2.character['name']}")
    print("-" * 40)
    
    # 3ターンだけテスト
    for turn in range(3):
        print(f"\n【ターン {turn + 1}】")
        
        result = await manager.run_turn()
        
        speaker = result.get('speaker', '不明')
        message = result.get('message', '')
        
        print(f"{speaker}:")
        print(f"  {message[:150]}..." if len(message) > 150 else f"  {message}")
        
        # 簡易チェック
        if turn > 0 and check_greeting_patterns(message):
            print("  ⚠️ 挨拶検出")
        if check_markdown_patterns(message):
            print("  ⚠️ マークダウン検出")
    
    print("\n✅ 一般キャラクターテスト完了")

if __name__ == "__main__":
    print("\n🚀 会話スタイル改善テストを開始します\n")
    
    try:
        # AI姉妹のテスト
        asyncio.run(test_conversation_style())
        
        # 一般キャラクターのテスト
        asyncio.run(test_general_characters())
        
        print("\n✨ 全テスト完了！")
        print("\n推奨事項:")
        print("1. 挨拶が続く場合は、prompt_templateの指示を強化")
        print("2. マークダウンが出る場合は、「普通の会話文で」を強調")
        print("3. Directorの介入も活用して矯正")
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()