# tests/test_refactoring.py
"""リファクタリング動作確認テスト"""

def test_ui_components():
    """UIコンポーネントのインポート確認"""
    try:
        from app.ui.components import DialogueUIComponents
        print("✅ UIコンポーネント: OK")
        return True
    except ImportError as e:
        print(f"❌ UIコンポーネント: {e}")
        return False

def test_controller():
    """Controllerの基本動作確認"""
    try:
        from app.core.dialogue_controller import DialogueController, DialogueConfig
        
        config = DialogueConfig(
            theme="テスト",
            agent1_name="high_school_girl_optimistic",
            agent2_name="office_worker_tired",
            max_turns=3
        )
        
        controller = DialogueController()
        controller.initialize_session(config)
        print("✅ Controller: OK")
        return True
    except Exception as e:
        print(f"❌ Controller: {e}")
        return False

def test_integration():
    """統合テスト"""
    try:
        import ollama
        from app.core.agent import Agent
        from app.core.director import AutonomousDirector
        
        # Ollama接続確認
        client = ollama.Client()
        client.list()
        print("✅ Ollama接続: OK")
        
        # エージェント作成
        agent = Agent(
            agent_id="test",
            character_type="high_school_girl_optimistic",
            model_name="qwen2.5:7b",
            ollama_client=client
        )
        print("✅ Agent作成: OK")
        
        # Director作成
        director = AutonomousDirector(client, "gemma3:4b")
        print("✅ Director作成: OK")
        
        return True
    except Exception as e:
        print(f"❌ 統合テスト: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("リファクタリング動作確認")
    print("=" * 50)
    
    all_pass = True
    all_pass &= test_ui_components()
    all_pass &= test_controller()
    all_pass &= test_integration()
    
    print("=" * 50)
    if all_pass:
        print("✅ すべてのテストが成功しました！")
    else:
        print("❌ 一部のテストが失敗しました")