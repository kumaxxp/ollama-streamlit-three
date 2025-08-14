#!/usr/bin/env python3
"""
統合テストスクリプト - 修正版
すべてのモジュールの接続を確認
"""
import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """インポートテスト"""
    print("=" * 50)
    print("1. インポートテスト")
    print("=" * 50)
    
    try:
        from app.core.dialogue_controller import DialogueController, DialogueConfig
        print("✅ dialogue_controller: OK")
    except ImportError as e:
        print(f"❌ dialogue_controller: {e}")
        return False
    
    try:
        from app.core.agent import Agent
        print("✅ agent: OK")
    except ImportError as e:
        print(f"❌ agent: {e}")
        return False
    
    try:
        from app.core.director import AutonomousDirector
        print("✅ director: OK")
    except ImportError as e:
        print(f"❌ director: {e}")
        return False
    
    try:
        from app.core.dialogue_manager import DialogueManager
        print("✅ dialogue_manager: OK")
    except ImportError as e:
        print(f"❌ dialogue_manager: {e}")
        return False
    
    return True

def test_ollama_connection():
    """Ollama接続テスト"""
    print("\n" + "=" * 50)
    print("2. Ollama接続テスト")
    print("=" * 50)
    
    try:
        import ollama
        client = ollama.Client()
        models = client.list()
        
        print(f"✅ Ollama接続: OK")
        print(f"   利用可能なモデル数: {len(models.get('models', []))}")
        
        # モデル一覧表示
        for model in models.get('models', [])[:5]:  # 最初の5個まで
            print(f"   - {model.get('name', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ollama接続: {e}")
        print("   Ollamaが起動していることを確認してください: ollama serve")
        return False

def test_controller_initialization():
    """Controller初期化テスト"""
    print("\n" + "=" * 50)
    print("3. Controller初期化テスト")
    print("=" * 50)
    
    try:
        from app.core.dialogue_controller import DialogueController, DialogueConfig
        
        # 設定作成
        config = DialogueConfig(
            theme="テストテーマ",
            agent1_name="high_school_girl_optimistic",
            agent2_name="office_worker_tired",
            max_turns=3,
            director_config={"model": "gemma3:4b"},
            model_params={"temperature": 0.7}
        )
        
        print(f"✅ 設定作成: OK")
        print(f"   テーマ: {config.theme}")
        print(f"   Agent1: {config.agent1_name}")
        print(f"   Agent2: {config.agent2_name}")
        
        # Controller初期化
        controller = DialogueController()
        controller.initialize_session(config)
        
        print(f"✅ Controller初期化: OK")
        print(f"   状態: {controller.get_state_summary()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Controller初期化: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_agent_creation():
    """Agent作成テスト"""
    print("\n" + "=" * 50)
    print("4. Agent作成テスト")
    print("=" * 50)
    
    try:
        from app.core.agent import Agent
        import ollama
        
        client = ollama.Client()
        
        # エージェント作成
        agent = Agent(
            agent_id="test_agent",
            character_type="high_school_girl_optimistic",
            model_name="qwen2.5:7b",
            temperature=0.7,
            ollama_client=client
        )
        
        print(f"✅ Agent作成: OK")
        print(f"   キャラクター: {agent.character.get('name', 'Unknown')}")
        print(f"   性格: {agent.character.get('personality', 'Unknown')[:50]}...")
        
        # セッションコンテキスト設定
        agent.set_session_context(
            theme="テストテーマ",
            goal="テスト目標",
            phase="exploration"
        )
        
        print(f"✅ セッションコンテキスト設定: OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent作成: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_director_creation():
    """Director作成テスト"""
    print("\n" + "=" * 50)
    print("5. Director作成テスト")
    print("=" * 50)
    
    try:
        from app.core.director import AutonomousDirector
        import ollama
        
        client = ollama.Client()
        
        # Director作成
        director = AutonomousDirector(
            ollama_client=client,
            model_name="gemma3:4b"
        )
        
        print(f"✅ Director作成: OK")
        print(f"   モデル: {director.model_name}")
        print(f"   温度: {director.temperature}")
        print(f"   現在フェーズ: {director.current_phase}")
        
        # フェーズ更新テスト
        director.update_phase(5)
        print(f"✅ フェーズ更新: OK")
        print(f"   更新後フェーズ: {director.current_phase}")
        
        return True
        
    except Exception as e:
        print(f"❌ Director作成: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_turn():
    """簡単なターン実行テスト"""
    print("\n" + "=" * 50)
    print("6. 簡単なターン実行テスト")
    print("=" * 50)
    
    try:
        from app.core.dialogue_controller import DialogueController, DialogueConfig
        
        # 設定
        config = DialogueConfig(
            theme="簡単なテスト",
            agent1_name="high_school_girl_optimistic",
            agent2_name="office_worker_tired",
            max_turns=1,
            director_config={"check_interval": 10},  # Directorを無効化
            model_params={"temperature": 0.5}
        )
        
        # Controller初期化
        controller = DialogueController()
        controller.initialize_session(config)
        
        print("⏳ ターン実行中...")
        
        # 1ターン実行
        events = []
        for event in controller.run_turn():
            events.append(event)
            print(f"   イベント: {event['type']}")
        
        if events:
            print(f"✅ ターン実行: OK")
            print(f"   イベント数: {len(events)}")
        else:
            print(f"⚠️ ターン実行: イベントなし")
        
        return True
        
    except Exception as e:
        print(f"❌ ターン実行: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト実行"""
    print("\n" + "=" * 60)
    print("   Ollama Streamlit Three - 統合テスト")
    print("=" * 60)
    
    tests = [
        ("インポート", test_imports),
        ("Ollama接続", test_ollama_connection),
        ("Controller初期化", test_controller_initialization),
        ("Agent作成", test_agent_creation),
        ("Director作成", test_director_creation),
        ("ターン実行", test_simple_turn)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name}で予期しないエラー: {e}")
            results.append((test_name, False))
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("   テスト結果サマリー")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("-" * 60)
    print(f"合計: {passed}/{total} テスト成功")
    
    if passed == total:
        print("\n🎉 すべてのテストが成功しました！")
        print("リファクタリングの準備が整いました。")
        return 0
    else:
        print(f"\n⚠️ {total - passed}個のテストが失敗しました。")
        print("上記のエラーを確認してください。")
        return 1

if __name__ == "__main__":
    exit(main())