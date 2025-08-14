#!/usr/bin/env python3
"""
キャラクター読み込みテスト
やなとあゆが正しく読み込まれるか確認
"""

import json
import os
import sys
from pathlib import Path

# パスを追加
sys.path.append(str(Path(__file__).parent))

def test_character_loading():
    """キャラクター読み込みテスト"""
    
    print("=" * 60)
    print("🎭 キャラクター読み込みテスト")
    print("=" * 60)
    
    # 1. characters.jsonの存在確認
    print("\n1. ファイル存在確認")
    print("-" * 40)
    
    possible_paths = [
        'config/characters.json',
        './config/characters.json',
        '../config/characters.json',
        str(Path(__file__).parent / 'config' / 'characters.json')
    ]
    
    found_path = None
    for path in possible_paths:
        exists = os.path.exists(path)
        print(f"  {path}: {'✅ 存在' if exists else '❌ なし'}")
        if exists and not found_path:
            found_path = path
    
    if not found_path:
        print("\n❌ characters.jsonが見つかりません！")
        print(f"現在のディレクトリ: {os.getcwd()}")
        return
    
    print(f"\n✅ 使用するファイル: {found_path}")
    
    # 2. JSON読み込みテスト
    print("\n2. JSON読み込みテスト")
    print("-" * 40)
    
    try:
        with open(found_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print("✅ JSONの読み込み成功")
    except Exception as e:
        print(f"❌ JSON読み込みエラー: {e}")
        return
    
    # 3. キャラクター存在確認
    print("\n3. AI姉妹キャラクター確認")
    print("-" * 40)
    
    if 'characters' not in data:
        print("❌ 'characters'キーが存在しません")
        return
    
    characters = data['characters']
    ai_sisters = [
        "AI-tuber-college_student_girl",  # やな
        "AI-tuber-high_school_girl"       # あゆ
    ]
    
    for char_key in ai_sisters:
        if char_key in characters:
            char_data = characters[char_key]
            print(f"✅ {char_key}")
            print(f"   名前: {char_data.get('name', '不明')}")
            print(f"   性格: {char_data.get('personality', '不明')[:50]}...")
        else:
            print(f"❌ {char_key} が見つかりません")
    
    # 4. 全キャラクターリスト
    print("\n4. 利用可能な全キャラクター")
    print("-" * 40)
    
    for i, (key, char) in enumerate(characters.items(), 1):
        name = char.get('name', '名前なし')
        print(f"  {i}. {key}")
        print(f"     → {name}")
    
    # 5. Agentクラスでのテスト
    print("\n5. Agentクラスでの読み込みテスト")
    print("-" * 40)
    
    try:
        from app.core.agent import Agent
        import ollama
        
        # やなのテスト
        agent_yana = Agent(
            agent_id="test_yana",
            character_type="AI-tuber-college_student_girl",
            model_name="qwen2.5:7b-instruct-q4_K_M",
            ollama_client=ollama.Client()
        )
        
        print(f"✅ やな: {agent_yana.character.get('name', '読み込み失敗')}")
        
        # あゆのテスト
        agent_ayu = Agent(
            agent_id="test_ayu",
            character_type="AI-tuber-high_school_girl",
            model_name="qwen2.5:7b-instruct-q4_K_M",
            ollama_client=ollama.Client()
        )
        
        print(f"✅ あゆ: {agent_ayu.character.get('name', '読み込み失敗')}")
        
    except Exception as e:
        print(f"❌ Agentクラステストエラー: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. 設定の妥当性チェック
    print("\n6. AI姉妹設定の妥当性チェック")
    print("-" * 40)
    
    for char_key in ai_sisters:
        if char_key in characters:
            char = characters[char_key]
            checks = {
                "name": "name" in char,
                "personality": "personality" in char,
                "speaking_style": "speaking_style" in char,
                "ai_characteristics": "ai_characteristics" in char,
                "prompt_template": "prompt_template" in char,
                "behavioral_patterns": "behavioral_patterns" in char
            }
            
            print(f"\n{char.get('name', char_key)}:")
            for check_name, exists in checks.items():
                print(f"  {check_name}: {'✅' if exists else '❌'}")

def fix_character_json():
    """characters.jsonの問題を自動修正"""
    print("\n" + "=" * 60)
    print("🔧 characters.json 自動修正")
    print("=" * 60)
    
    # ディレクトリ作成
    os.makedirs('config', exist_ok=True)
    
    # 最小限のAI姉妹設定
    min_config = {
        "characters": {
            "AI-tuber-college_student_girl": {
                "name": "やな（AI姉・明るいお姉ちゃんAI）",
                "personality": "明るく元気なAI姉",
                "speaking_style": "カジュアルでフレンドリー",
                "background": "AI姉妹の姉",
                "values": "妹のあゆとの絆",
                "behavioral_patterns": {
                    "greeting": ["やっほー！"],
                    "agreement": ["せやな！"],
                    "disagreement": ["えー、でも..."]
                },
                "prompt_template": "あなたは{name}です。"
            },
            "AI-tuber-high_school_girl": {
                "name": "あゆ（AI妹・冷静なツッコミAI）",
                "personality": "冷静で理知的なAI妹",
                "speaking_style": "丁寧で落ち着いた口調",
                "background": "AI姉妹の妹",
                "values": "姉のやなのサポート",
                "behavioral_patterns": {
                    "greeting": ["こんにちは"],
                    "agreement": ["論理的に正しいです"],
                    "disagreement": ["エラーを検出しました"]
                },
                "prompt_template": "あなたは{name}です。"
            }
        }
    }
    
    # バックアップ作成
    if os.path.exists('config/characters.json'):
        import shutil
        backup_path = 'config/characters.json.backup'
        shutil.copy('config/characters.json', backup_path)
        print(f"✅ バックアップ作成: {backup_path}")
    
    # 修正版を保存
    with open('config/characters.json.fixed', 'w', encoding='utf-8') as f:
        json.dump(min_config, f, ensure_ascii=False, indent=2)
    
    print("✅ 修正版を作成: config/characters.json.fixed")
    print("\n使用方法:")
    print("  cp config/characters.json.fixed config/characters.json")

if __name__ == "__main__":
    print("\n🚀 キャラクター読み込みテストを開始\n")
    
    test_character_loading()
    
    # 問題があれば修正を提案
    if not os.path.exists('config/characters.json'):
        print("\n⚠️ characters.jsonが存在しません")
        choice = input("\n修正ファイルを作成しますか？ (y/n): ")
        if choice.lower() == 'y':
            fix_character_json()
    
    print("\n✨ テスト完了！")