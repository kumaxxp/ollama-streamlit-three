"""
Advanced Dialogue System - Refactored UI
会話履歴をすべて表示する改良版
"""
import streamlit as st
import json
from pathlib import Path
import sys
from datetime import datetime

# パス設定
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.core.dialogue_controller import DialogueController, DialogueConfig
from app.ui.components import (
    DialogueUIComponents, 
    ParameterControls, 
    DialogueDisplay
)
from app.ui.streamlit_helpers import (
    get_available_models,
    get_character_options,
    get_theme_options,
    check_ollama_connection,
    show_connection_error,
    get_character_icon
)

# ページ設定
st.set_page_config(
    page_title="Advanced Dialogue (Refactored)",
    page_icon="🎭",
    layout="wide"
)

# Ollama接続確認
if not check_ollama_connection():
    show_connection_error()
    st.stop()

# コンポーネント初期化
ui = DialogueUIComponents()
params = ParameterControls()

# セッション状態初期化
if "controller" not in st.session_state:
    st.session_state.controller = None
if "config" not in st.session_state:
    st.session_state.config = None
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "dialogue_display" not in st.session_state:
    st.session_state.dialogue_display = DialogueDisplay()
if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0
if "dialogue_history" not in st.session_state:
    st.session_state.dialogue_history = []  # 対話履歴を保存

# ヘッダー
ui.render_header("🎭 Advanced Dialogue System (Refactored)")

# サイドバー設定
with st.sidebar:
    st.header("⚙️ 設定")
    
    # テーマ選択
    st.subheader("📝 テーマ")
    theme_options = get_theme_options()
    selected_theme = st.selectbox("テーマを選択", theme_options)
    
    if selected_theme == "カスタム（下に入力）":
        theme = st.text_input("カスタムテーマ", placeholder="議論したいテーマを入力")
        if not theme:
            theme = "自由討論"
    else:
        theme = selected_theme
    
    st.divider()
    
    # キャラクター選択
    st.subheader("👥 キャラクター")
    char_options = get_character_options()
    
    agent1_key = st.selectbox(
        "Agent 1",
        options=list(char_options.keys()),
        format_func=lambda x: char_options[x],
        index=0
    )
    
    agent2_key = st.selectbox(
        "Agent 2",
        options=list(char_options.keys()),
        format_func=lambda x: char_options[x],
        index=1
    )
    
    st.divider()
    
    # モデル設定
    st.subheader("🤖 モデル設定")
    
    models = get_available_models()
    
    # デフォルトモデルの設定
    default_agent_model = "qwen2.5:7b-instruct-q4_K_M"
    default_director_model = "gemma3:4b"
    
    # モデルリストにデフォルトがない場合は最初のモデルを使用
    if default_agent_model not in models:
        default_agent_model = models[0] if models else "qwen:7b"
    
    if default_director_model not in models:
        default_director_model = models[1] if len(models) > 1 else models[0]
    
    agent_model = st.selectbox(
        "エージェントモデル",
        options=models,
        index=models.index(default_agent_model) if default_agent_model in models else 0,
        help="対話エージェント用のモデル"
    )
    
    director_model = st.selectbox(
        "Directorモデル",
        options=models,
        index=models.index(default_director_model) if default_director_model in models else min(1, len(models)-1),
        help="監督AI用のモデル"
    )
    
    # モデル情報表示
    if agent_model:
        if "qwen2.5:7b-instruct-q4_K_M" in agent_model:
            st.success("✅ 推奨モデル（日本語対話に最適）")
        elif "qwen" in agent_model.lower():
            st.info("✓ Qwenモデル（日本語対応）")
    
    if director_model:
        if "gemma3:4b" in director_model or "gemma2:2b" in director_model:
            st.success("✅ Director推奨（高速判断）")
    
    st.divider()
    
    # パラメータ設定
    with st.expander("詳細設定", expanded=False):
        max_turns = st.slider("最大ターン数", 5, 30, 20)
        
        col1, col2 = st.columns(2)
        with col1:
            agent_temp = st.slider("Agent温度", 0.1, 1.0, 0.7, 0.1)
        with col2:
            director_temp = st.slider("Director温度", 0.1, 1.0, 0.3, 0.1)
        
        check_interval = st.number_input(
            "Director介入間隔",
            min_value=1,
            max_value=10,
            value=2,
            help="何ターンごとにDirectorが分析するか"
        )
        
        show_analysis = st.checkbox("Director分析を表示", value=False)
        auto_scroll = st.checkbox("自動スクロール", value=True, help="新しい発言に自動でスクロール")

# メインエリア - コントロールボタン
col1, col2, col3, col4 = st.columns(4)

with col1:
    start_button = st.button(
        "🎬 開始",
        type="primary",
        disabled=st.session_state.is_running,
        use_container_width=True
    )

with col2:
    stop_button = st.button(
        "⏸️ 停止",
        disabled=not st.session_state.is_running,
        use_container_width=True
    )

with col3:
    reset_button = st.button(
        "🔄 リセット",
        use_container_width=True
    )

with col4:
    # ターン数表示
    if st.session_state.controller and st.session_state.controller.state:
        current_max = st.session_state.config.max_turns if st.session_state.config else max_turns
        st.metric("ターン", f"{st.session_state.turn_count}/{current_max}")

# ボタンアクション処理
if start_button:
    # Controller設定を作成してセッション状態に保存
    st.session_state.config = DialogueConfig(
        theme=theme,
        agent1_name=agent1_key,
        agent2_name=agent2_key,
        max_turns=max_turns,
        director_config={
            "model": director_model,
            "temperature": director_temp,
            "check_interval": check_interval
        },
        model_params={
            "model": agent_model,
            "temperature": agent_temp
        }
    )
    
    # Controller初期化
    st.session_state.controller = DialogueController()
    st.session_state.controller.initialize_session(st.session_state.config)
    st.session_state.is_running = True
    st.session_state.turn_count = 0
    st.session_state.dialogue_history = []  # 履歴をクリア
    st.session_state.dialogue_display.clear()
    
    # 開始メッセージを履歴に追加
    st.session_state.dialogue_history.append({
        "type": "system",
        "content": f"🎭 対話開始: 「{theme}」",
        "timestamp": datetime.now().isoformat()
    })
    
    st.success(f"✅ 対話を開始しました！使用モデル: {agent_model}")
    st.rerun()

if stop_button:
    if st.session_state.controller:
        st.session_state.controller.stop()
    st.session_state.is_running = False
    
    # 停止メッセージを履歴に追加
    st.session_state.dialogue_history.append({
        "type": "system",
        "content": "⏸️ 対話を一時停止しました",
        "timestamp": datetime.now().isoformat()
    })
    
    st.info("⏸️ 対話を停止しました")

if reset_button:
    if st.session_state.controller:
        st.session_state.controller.reset()
    st.session_state.controller = None
    st.session_state.config = None
    st.session_state.is_running = False
    st.session_state.turn_count = 0
    st.session_state.dialogue_history = []  # 履歴をクリア
    st.session_state.dialogue_display.clear()
    st.info("🔄 リセットしました")
    st.rerun()

# 対話履歴表示エリア
st.markdown("---")
st.subheader("💬 対話履歴")

# 履歴表示用のコンテナ
dialogue_container = st.container()

# 既存の履歴をすべて表示
with dialogue_container:
    for entry in st.session_state.dialogue_history:
        if entry["type"] == "system":
            # システムメッセージ
            st.info(entry["content"])
        
        elif entry["type"] == "agent":
            # エージェントの発言
            char_name = entry.get("name", "Unknown")
            icon = entry.get("icon", "👤")
            message = entry.get("content", "")
            turn = entry.get("turn", 0)
            
            with st.chat_message("assistant", avatar=icon):
                st.markdown(f"**{char_name}** (Turn {turn})")
                st.write(message)
        
        elif entry["type"] == "director":
            # Director介入
            if show_analysis or entry.get("important", False):
                with st.expander("🎬 Director介入", expanded=False):
                    st.write(entry["content"])

# 自動スクロール用の空要素
if 'auto_scroll' in locals() and auto_scroll:
    st.empty()

# 対話実行
if st.session_state.is_running and st.session_state.controller and st.session_state.config:
    # 1ターン実行
    with st.spinner(f"Turn {st.session_state.turn_count + 1} 実行中..."):
        events = []
        
        try:
            # ターン実行
            for event in st.session_state.controller.run_turn():
                events.append(event)
                
                # イベント処理
                if event["type"] == "agent_response":
                    agent_name = event["data"]["agent"]
                    response = event["data"]["response"]
                    
                    # キャラクター名とアイコンを取得
                    if agent_name in st.session_state.controller.agents:
                        char_name = st.session_state.controller.agents[agent_name].character.get('name', agent_name)
                        icon = get_character_icon(char_name)
                    else:
                        char_name = agent_name
                        icon = "👤"
                    
                    # 履歴に追加
                    st.session_state.dialogue_history.append({
                        "type": "agent",
                        "name": char_name,
                        "icon": icon,
                        "content": response,
                        "turn": st.session_state.turn_count + 1,
                        "timestamp": datetime.now().isoformat()
                    })
                
                elif event["type"] == "director_intervention":
                    # Director介入を履歴に追加
                    st.session_state.dialogue_history.append({
                        "type": "director",
                        "content": event['data']['message'],
                        "important": True,
                        "timestamp": datetime.now().isoformat()
                    })
                
                elif event["type"] == "director_analysis":
                    # Director分析を履歴に追加（詳細設定でONの場合のみ表示）
                    if show_analysis:
                        st.session_state.dialogue_history.append({
                            "type": "director",
                            "content": f"分析結果: {json.dumps(event['data'], ensure_ascii=False, indent=2)}",
                            "important": False,
                            "timestamp": datetime.now().isoformat()
                        })
                
                elif event["type"] == "turn_complete":
                    st.session_state.turn_count = event["data"]["turn_count"]
            
            # 最大ターン到達確認
            if st.session_state.turn_count >= st.session_state.config.max_turns:
                st.session_state.is_running = False
                
                # 完了メッセージを履歴に追加
                st.session_state.dialogue_history.append({
                    "type": "system",
                    "content": f"✅ 対話が完了しました！（全{st.session_state.turn_count}ターン）",
                    "timestamp": datetime.now().isoformat()
                })
                
                st.success("✅ 対話が完了しました！")
            else:
                # 次のターンのために再実行
                st.rerun()
                
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
            
            # エラーメッセージを履歴に追加
            st.session_state.dialogue_history.append({
                "type": "system",
                "content": f"❌ エラー: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
            
            import traceback
            with st.expander("エラー詳細", expanded=False):
                st.code(traceback.format_exc())
            
            st.session_state.is_running = False

# 統計情報とエクスポート
if st.session_state.controller and st.session_state.controller.state:
    with st.sidebar:
        st.divider()
        st.subheader("📊 統計")
        
        state_summary = st.session_state.controller.get_state_summary()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("総ターン数", state_summary.get("turn_count", 0))
        with col2:
            avg_time = state_summary.get("avg_response_time", 0)
            if avg_time > 0:
                st.metric("平均応答時間", f"{avg_time:.2f}秒")
        
        # 対話履歴の統計
        if st.session_state.dialogue_history:
            agent_messages = [h for h in st.session_state.dialogue_history if h["type"] == "agent"]
            director_messages = [h for h in st.session_state.dialogue_history if h["type"] == "director"]
            
            st.metric("総発言数", len(agent_messages))
            if director_messages:
                st.metric("Director介入", len(director_messages))
        
        st.divider()
        
        # エクスポート機能
        st.subheader("💾 保存")
        
        # 履歴をエクスポート
        if st.button("📥 対話履歴をダウンロード", use_container_width=True):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # エクスポートデータを作成
            export_data = {
                "metadata": {
                    "theme": st.session_state.config.theme if st.session_state.config else theme,
                    "agents": [agent1_key, agent2_key],
                    "models": {
                        "agent": agent_model,
                        "director": director_model
                    },
                    "total_turns": st.session_state.turn_count,
                    "timestamp": timestamp
                },
                "dialogue": st.session_state.dialogue_history,
                "statistics": state_summary if st.session_state.controller else {}
            }
            
            json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
            
            st.download_button(
                label="💾 JSONファイルをダウンロード",
                data=json_str,
                file_name=f"dialogue_{timestamp}.json",
                mime="application/json",
                use_container_width=True
            )
        
        # 履歴をクリア
        if st.button("🗑️ 履歴をクリア", use_container_width=True):
            st.session_state.dialogue_history = []
            st.info("履歴をクリアしました")
            st.rerun()

# デバッグ情報（開発時のみ）
with st.sidebar:
    with st.expander("🔧 デバッグ情報", expanded=False):
        st.code(f"""
Controller: {st.session_state.controller is not None}
Config: {st.session_state.config is not None}
Running: {st.session_state.is_running}
Turn: {st.session_state.turn_count}
History Length: {len(st.session_state.dialogue_history)}
Agent Model: {agent_model if 'agent_model' in locals() else 'Not set'}
Director Model: {director_model if 'director_model' in locals() else 'Not set'}
        """)

# フッター
st.markdown("---")
st.caption("Advanced Dialogue System v2.0 - Refactored Edition")
st.caption("✅ 会話履歴完全表示対応 | Controller統合完了 | UIコンポーネント分離")