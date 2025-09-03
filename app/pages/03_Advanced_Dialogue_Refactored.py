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
from app.core.model_utils import ModelManager

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
    mm = ModelManager()
    
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
        dm = str(director_model)
        if "gemma3:4b" in dm or "gemma2:2b" in dm:
            st.success("✅ Director推奨（高速判断）")
    
    st.divider()
    
    # パラメータ設定
    with st.expander("詳細設定", expanded=False):
        max_turns = st.slider("最大ターン数", 5, 30, 20)
        
        # Agent温度（個別設定）
        default_agent_temp = mm.get_recommended_temperature(agent_model, use_case="agent")
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            agent1_temp = st.slider("Agent1温度", 0.1, 1.0, float(default_agent_temp), 0.1)
        with col_a2:
            agent2_temp = st.slider("Agent2温度", 0.1, 1.0, float(default_agent_temp), 0.1)

        # Director温度（個別）
        default_director_temp = mm.get_recommended_temperature(director_model, use_case="director")
        director_temp = st.slider("Director温度", 0.1, 1.0, float(default_director_temp), 0.1)
        
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
            # 後方互換のためtemperatureも残す（未使用時のフォールバック）
            "temperature": float(agent1_temp),
            # 個別温度
            "agent1_temperature": float(agent1_temp),
            "agent2_temperature": float(agent2_temp)
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
        
        elif entry["type"] == "agent" or entry["type"] == "agent_error":
            # エージェントの発言
            char_name = entry.get("name", "Unknown")
            icon = entry.get("icon", "👤")
            message = entry.get("content", "")
            turn = entry.get("turn", 0)
            
            # エラーメッセージは強調表示
            if entry["type"] == "agent_error":
                with st.container():
                    st.error(f"⚠️ {char_name} (Turn {turn}) - 応答にエラーが発生しました")
                    st.write(message)
                    if entry.get('detail'):
                        with st.expander("詳細エラー情報", expanded=False):
                            st.code(entry.get('detail'))
            else:
                with st.chat_message("assistant", avatar=icon):
                    st.markdown(f"**{char_name}** (Turn {turn})")
                    st.write(message)

        elif entry["type"] == "agent_prompt":
            # システム/ユーザープロンプトを表示（レスポンス前に入る）
            st.markdown(f"**{entry['icon']} {entry['name']} のプロンプト（Turn {entry['turn']})**")
            with st.expander("System Prompt", expanded=False):
                st.code(entry.get("system_prompt", ""))
            with st.expander("User Prompt", expanded=False):
                st.code(entry.get("user_prompt", ""))
        
        elif entry["type"] == "director":
            # Director介入（詳細表示を併置）
            dbg = entry.get("debug") or {}
            with st.expander("🎬 Director介入", expanded=entry.get("important", False)):
                # 介入メッセージ
                st.write(entry["content"])
                # デバッグ（検出・検証の可視化）
                if dbg:
                    colL, colR = st.columns(2)
                    with colL:
                        st.caption("🔎 検出候補 (heuristic/LLM)")
                        st.code(json.dumps({
                            "heuristic": dbg.get("heuristic_entities"),
                            "llm": dbg.get("llm_entities"),
                        }, ensure_ascii=False, indent=2))
                        # 追加: 軽量ファクトチェック（claims / findings）
                        try:
                            if isinstance(dbg.get("light_factcheck"), dict):
                                lf = dbg.get("light_factcheck")
                                st.caption("🧪 軽量ファクトチェック")
                                st.code(json.dumps(lf, ensure_ascii=False, indent=2))
                        except Exception:
                            pass
                        # 追加: 強化ファクトチェック（CoVe/FEVER/SelfConsistency）
                        try:
                            if isinstance(dbg.get("strong_factcheck"), dict):
                                sf = dbg.get("strong_factcheck")
                                st.caption("🧪 強化ファクトチェック (CoVe→FEVER→SelfConsistency)")
                                st.code(json.dumps(sf, ensure_ascii=False, indent=2))
                        except Exception:
                            pass
                        if dbg.get("holistic_text"):
                            st.caption("🧠 ホリスティックレビュー(テキスト)")
                            st.write(dbg.get("holistic_text"))
                        # 2回目の再レビュー情報（存在すれば）
                        if isinstance(entry.get("content"), str) and 'challenge_and_verify' in str(entry.get("content")):
                            st.caption("🧠 再レビュー(証拠取り込み)")
                            st.write("証拠を取り込んだ再レビューが実行され、pushbackや指示が強化されています。")
                    with colR:
                        st.caption("✅ 候補と検証")
                        st.code(json.dumps({
                            "selected": dbg.get("selected_candidate"),
                            "verification": dbg.get("verification"),
                            "all_candidates": dbg.get("all_candidates"),
                            "verifications": dbg.get("verifications"),
                        }, ensure_ascii=False, indent=2))
                        # 追加: Web検索の可視化（MCP/Wikipedia）
                        try:
                            has_search = bool(dbg.get("research") or dbg.get("wiki_snippets"))
                        except Exception:
                            has_search = False
                        if has_search:
                            st.caption("🌐 Web検索ログ（Director実行）")
                            # 1) その場検索 research（単発）
                            if isinstance(dbg.get("research"), list) and dbg.get("research"):
                                st.markdown("- その場検証（キーワード指定）")
                                try:
                                    for r in dbg.get("research")[:3]:
                                        if isinstance(r, dict):
                                            q = r.get("query")
                                            v = r.get("verdict")
                                            u = r.get("evidence")
                                            ex = r.get("evidence_text")
                                            st.write(f"• 検索: {q} / 判定: {v}")
                                            if u:
                                                st.write(f"  URL: {u}")
                                            if ex:
                                                st.write("  要約: " + str(ex)[:240])
                                except Exception:
                                    pass
                            # 2) Wikipediaスニペット（複数候補）
                            if isinstance(dbg.get("wiki_snippets"), list) and dbg.get("wiki_snippets"):
                                st.markdown("- Wikipediaスニペット")
                                try:
                                    for s in dbg.get("wiki_snippets")[:4]:
                                        if isinstance(s, dict):
                                            q = s.get("query")
                                            t = s.get("title")
                                            u = s.get("url")
                                            ex = s.get("excerpt")
                                            header = f"• 検索: {q} → 候補: {t}" if q else f"• 候補: {t}"
                                            st.write(header)
                                            if ex:
                                                st.write("  要約: " + str(ex)[:240])
                                            if u:
                                                st.write(f"  URL: {u}")
                                except Exception:
                                    pass
                        # 3) 地理や作品検証の補助情報
                        try:
                            if isinstance(dbg.get("geo"), dict) and dbg.get("geo"):
                                st.caption("🗺️ 地理チェック")
                                st.code(json.dumps(dbg.get("geo"), ensure_ascii=False, indent=2))
                            if isinstance(dbg.get("works_detected"), list) and dbg.get("works_detected"):
                                st.caption("📚 作品検証")
                                st.code(json.dumps(dbg.get("works_detected"), ensure_ascii=False, indent=2))
                        except Exception:
                            pass
        elif entry["type"] == "director_analysis_event":
            # 任意のタイミングでのDirector分析スナップショット
            if show_analysis:
                with st.expander("🧪 Director分析スナップショット", expanded=False):
                    st.code(json.dumps(entry["content"], ensure_ascii=False, indent=2))

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
                if event["type"] == "agent_prompts":
                    agent_name = event["data"]["agent"]
                    system_prompt = event["data"].get("system_prompt")
                    user_prompt = event["data"].get("user_prompt")

                    # 表示用にキャラクター名を取得
                    if agent_name in st.session_state.controller.agents:
                        char_name = st.session_state.controller.agents[agent_name].character.get('name', agent_name)
                        icon = get_character_icon(char_name)
                    else:
                        char_name = agent_name
                        icon = "👤"

                    # 履歴にプロンプト情報を追加（表示はレスポンスの直前に行う）
                    st.session_state.dialogue_history.append({
                        "type": "agent_prompt",
                        "name": char_name,
                        "icon": icon,
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "turn": st.session_state.turn_count + 1,
                        "timestamp": datetime.now().isoformat()
                    })

                    continue

                if event["type"] == "agent_response":
                    agent_name = event["data"]["agent"]
                    response = event["data"]["response"]
                    is_error = event["data"].get('error', False)
                    detail = event["data"].get('detail') if is_error else None

                    # キャラクター名とアイコンを取得
                    if agent_name in st.session_state.controller.agents:
                        char_name = st.session_state.controller.agents[agent_name].character.get('name', agent_name)
                        icon = get_character_icon(char_name)
                    else:
                        char_name = agent_name
                        icon = "👤"

                    # 履歴に追加
                    entry_type = "agent_error" if is_error else "agent"
                    st.session_state.dialogue_history.append({
                        "type": entry_type,
                        "name": char_name,
                        "icon": icon,
                        "content": response,
                        "turn": st.session_state.turn_count + 1,
                        "timestamp": datetime.now().isoformat(),
                        "detail": detail
                    })
                
                elif event["type"] == "director_intervention":
                    # Director介入（詳細と併せて可視化）
                    intervention = event["data"]
                    dbg = intervention.get("director_debug") if isinstance(intervention, dict) else None
                    st.session_state.dialogue_history.append({
                        "type": "director",
                        "content": intervention.get('message', ''),
                        "important": True,
                        "timestamp": datetime.now().isoformat(),
                        "debug": dbg,
                    })
                
                elif event["type"] == "director_analysis":
                    # Director分析結果を履歴に常に保存（UIで並置表示に使用）
                    analysis = event["data"]
                    st.session_state.dialogue_history.append({
                        "type": "director_analysis_event",
                        "content": analysis,
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