"""
Advanced Dialogue System - Streamlit UI
Director制御による高品質対話生成
"""

import streamlit as st
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

# パスを追加（core モジュールをインポートするため）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dialogue_manager import DialogueManager
from core.agent import Agent
from core.director import Director

# ページ設定
st.set_page_config(
    page_title="Advanced Dialogue System",
    page_icon="🎯",
    layout="wide"
)

# ============ ヘルパー関数 ============

def load_characters() -> Dict:
    """利用可能なキャラクターを読み込み"""
    try:
        with open("config/characters.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"キャラクター設定の読み込みエラー: {e}")
        return {}

def get_available_models():
    """利用可能なモデルを取得"""
    try:
        import ollama
        models_response = ollama.list()
        available_models = []
        
        if hasattr(models_response, 'models'):
            for model in models_response.models:
                if hasattr(model, 'model'):
                    available_models.append(model.model)
                elif hasattr(model, 'name'):
                    available_models.append(model.name)
        elif isinstance(models_response, dict) and 'models' in models_response:
            for model in models_response['models']:
                if isinstance(model, dict) and 'name' in model:
                    available_models.append(model['name'])
                    
        if not available_models:
            available_models = ["qwen2.5:7b", "gemma3:4b", "llama3.2:3b"]
            
        return available_models
    except Exception as e:
        st.warning(f"モデル取得エラー: {e}")
        return ["qwen2.5:7b", "gemma3:4b"]

def format_quality_score(score: Dict) -> str:
    """品質スコアをフォーマット"""
    overall = score.get("overall_score", 0)
    if overall >= 8:
        color = "green"
        emoji = "🌟"
    elif overall >= 6:
        color = "orange"
        emoji = "✨"
    else:
        color = "red"
        emoji = "⚠️"
    
    return f":{color}[{emoji} {overall:.1f}/10]"

def display_dialogue_turn(turn_data: Dict):
    """対話ターンを表示"""
    for exchange in turn_data.get("exchanges", []):
        speaker = exchange["speaker"]
        content = exchange["content"]
        instruction = exchange.get("instruction", "")
        
        # 話者に応じたアバター
        avatar = "🎭" if "agent1" in speaker.lower() else "🔬"
        
        with st.chat_message("assistant", avatar=avatar):
            st.markdown(f"**{speaker}**")
            st.write(content)
            
            # Director指示を表示（デバッグモード時）
            if st.session_state.get("show_director_instructions", False):
                with st.expander("Director指示", expanded=False):
                    st.caption(f"📝 {instruction}")

# ============ メインUI ============

st.title("🎯 Advanced Dialogue System")
st.caption("Director AIが管理する高品質対話生成システム")

# サイドバー設定
with st.sidebar:
    st.header("⚙️ 設定")
    
    # キャラクター選択
    characters = load_characters()
    character_names = list(characters.keys())
    
    if character_names:
        st.subheader("🎭 キャラクター選択")
        
        col1, col2 = st.columns(2)
        with col1:
            agent1_char = st.selectbox(
                "エージェント1",
                character_names,
                index=0,
                key="agent1_character"
            )
            if agent1_char:
                st.caption(characters[agent1_char]["personality"]["base"])
        
        with col2:
            agent2_char = st.selectbox(
                "エージェント2",
                character_names,
                index=min(1, len(character_names)-1),
                key="agent2_character"
            )
            if agent2_char:
                st.caption(characters[agent2_char]["personality"]["base"])
    
    # モデル選択
    st.subheader("🤖 モデル設定")
    available_models = get_available_models()
    
    agent_model = st.selectbox(
        "エージェントモデル",
        available_models,
        index=0,
        key="agent_model"
    )
    
    director_model = st.selectbox(
        "Directorモデル",
        available_models,
        index=0,
        key="director_model"
    )
    
    # 詳細設定
    with st.expander("🔧 詳細設定", expanded=False):
        max_turns = st.slider("最大ターン数", 5, 30, 10, key="max_turns")
        agent_temp = st.slider("エージェント温度", 0.1, 1.0, 0.7, 0.1)
        director_temp = st.slider("Director温度", 0.1, 0.7, 0.3, 0.1)
        
        st.divider()
        show_director = st.checkbox(
            "Director指示を表示",
            value=False,
            key="show_director_instructions"
        )
        auto_save = st.checkbox("自動保存", value=True)

# メインエリア
tabs = st.tabs(["🎬 対話生成", "📊 分析", "📚 履歴", "📖 説明"])

# ============ タブ1: 対話生成 ============
with tabs[0]:
    # テーマ入力
    st.subheader("📝 テーマ設定")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        preset_themes = [
            "AIの意識と感情について",
            "持続可能な社会の実現方法",
            "教育の未来とテクノロジー",
            "芸術におけるオリジナリティとは",
            "人間の自由意志は存在するか",
            "カスタム（自由入力）"
        ]
        
        selected_preset = st.selectbox("テーマを選択", preset_themes)
        
        if selected_preset == "カスタム（自由入力）":
            theme = st.text_input("テーマを入力", placeholder="議論したいテーマを入力してください")
        else:
            theme = selected_preset
    
    with col2:
        st.write("")  # スペーサー
        st.write("")  # スペーサー
        start_btn = st.button(
            "🚀 対話開始",
            type="primary",
            disabled=not theme or theme == "カスタム（自由入力）",
            use_container_width=True
        )
    
    # 対話生成処理
    if start_btn:
        # セッション状態を初期化
        st.session_state.dialogue_manager = DialogueManager(
            theme=theme,
            agent1_config={
                "character": st.session_state.agent1_character,
                "model": st.session_state.agent_model,
                "temperature": agent_temp
            },
            agent2_config={
                "character": st.session_state.agent2_character,
                "model": st.session_state.agent_model,
                "temperature": agent_temp
            },
            director_config={
                "model": st.session_state.director_model,
                "temperature": director_temp
            },
            max_turns=st.session_state.max_turns
        )
        
        st.session_state.dialogue_history = []
        st.session_state.turn_results = []
        
        # プログレスバー
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 対話コンテナ
        dialogue_container = st.container()
        
        # 対話生成ループ
        for turn in range(st.session_state.max_turns):
            progress = (turn + 1) / st.session_state.max_turns
            progress_bar.progress(progress)
            
            # ストリーミング対話生成
            current_speaker = None
            current_message = ""
            message_placeholder = None
            
            for event in st.session_state.dialogue_manager.run_turn_streaming(
                first_speaker="agent1" if turn % 2 == 0 else "agent2"
            ):
                event_type = event.get("type")
                
                if event_type == "status":
                    status_text.text(event["message"])
                    
                elif event_type == "analysis":
                    # Director分析結果（デバッグ用）
                    if st.session_state.show_director_instructions:
                        with dialogue_container:
                            with st.expander("🔍 Director分析", expanded=False):
                                st.json(event["data"])
                                
                elif event_type == "response_chunk":
                    # ストリーミング応答
                    speaker = event["speaker"]
                    chunk = event["content"]
                    
                    if speaker != current_speaker:
                        # 新しい話者
                        current_speaker = speaker
                        current_message = chunk
                        
                        avatar = "🎭" if "agent1" in speaker.lower() else "🔬"
                        with dialogue_container:
                            with st.chat_message("assistant", avatar=avatar):
                                st.markdown(f"**{speaker}**")
                                message_placeholder = st.empty()
                                message_placeholder.write(current_message)
                    else:
                        # 既存メッセージに追加
                        current_message += chunk
                        if message_placeholder:
                            message_placeholder.write(current_message)
                            
                elif event_type == "response_complete":
                    # 応答完了
                    st.session_state.dialogue_history.append({
                        "speaker": event["speaker"],
                        "content": event["content"]
                    })
                    
                elif event_type == "phase_transition":
                    # フェーズ移行
                    with dialogue_container:
                        st.info(f"📊 フェーズ移行: {event['new_phase']}")
                        
                elif event_type == "turn_complete":
                    # ターン完了
                    pass
            
            # 中断チェック
            if st.button("⏸️ 中断", key=f"stop_btn_{turn}"):
                st.warning("対話を中断しました")
                break
        
        progress_bar.progress(1.0)
        status_text.text("✅ 対話生成完了！")
        
        # 自動保存
        if auto_save and st.session_state.dialogue_manager:
            filepath = st.session_state.dialogue_manager.save_dialogue()
            st.success(f"💾 自動保存完了: {filepath}")
        
        # 手動保存ボタン
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            if st.button("💾 保存", key="save_final"):
                if st.session_state.dialogue_manager:
                    filepath = st.session_state.dialogue_manager.save_dialogue()
                    st.success(f"保存完了: {filepath}")
        
        with col2:
            if st.button("🔄 新規対話", key="new_dialogue"):
                st.rerun()

# ============ タブ2: 分析 ============
with tabs[1]:
    st.subheader("📊 対話分析")
    
    if "dialogue_manager" in st.session_state and st.session_state.dialogue_manager:
        dm = st.session_state.dialogue_manager
        summary = dm.get_summary()
        
        # 基本統計
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("総ターン数", summary.get("total_turns", 0))
        with col2:
            st.metric("最終フェーズ", summary.get("final_phase", ""))
        with col3:
            st.metric("総発言数", summary.get("dialogue_length", 0))
        with col4:
            director_stats = summary.get("director_statistics", {})
            st.metric("平均深度", f"{director_stats.get('average_depth', 0):.1f}")
        
        # Director統計
        st.divider()
        st.subheader("🎯 Director統計")
        
        if director_stats:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("総介入回数", director_stats.get("total_interventions", 0))
            with col2:
                st.metric("平均生産性", f"{director_stats.get('average_productivity', 0):.2f}")
            with col3:
                st.metric("分析回数", director_stats.get("analysis_count", 0))
        
        # フェーズ分析
        st.divider()
        st.subheader("📈 フェーズ進行")
        
        phase_turns = summary.get("phase_turns", {})
        if phase_turns:
            st.bar_chart(phase_turns)
        
        # エージェント情報
        st.divider()
        st.subheader("🎭 エージェント情報")
        
        col1, col2 = st.columns(2)
        with col1:
            agent1_info = summary.get("agent1", {})
            st.write(f"**{agent1_info.get('name', '')}**")
            st.caption(f"タイプ: {agent1_info.get('type', '')}")
            st.caption(f"発言数: {agent1_info.get('response_count', 0)}")
        
        with col2:
            agent2_info = summary.get("agent2", {})
            st.write(f"**{agent2_info.get('name', '')}**")
            st.caption(f"タイプ: {agent2_info.get('type', '')}")
            st.caption(f"発言数: {agent2_info.get('response_count', 0)}")
        
    else:
        st.info("対話を生成すると分析結果が表示されます")

# ============ タブ3: 履歴 ============
with tabs[2]:
    st.subheader("📚 保存済み対話")
    
    dialogue_dir = os.path.join("data", "dialogues")
    
    if os.path.exists(dialogue_dir):
        files = sorted(
            [f for f in os.listdir(dialogue_dir) if f.endswith(".json")],
            reverse=True
        )
        
        if files:
            selected_file = st.selectbox("履歴を選択", files)
            
            if st.button("📂 読み込み", key="load_history"):
                try:
                    filepath = os.path.join(dialogue_dir, selected_file)
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    # 基本情報表示
                    st.write(f"**テーマ**: {data.get('theme', '')}")
                    st.write(f"**生成日時**: {data.get('timestamp', '')}")
                    st.write(f"**総ターン数**: {data.get('total_turns', 0)}")
                    
                    # エージェント情報
                    agents = data.get("agents", {})
                    col1, col2 = st.columns(2)
                    with col1:
                        agent1 = agents.get("agent1", {})
                        st.write(f"**Agent 1**: {agent1.get('name', '')}")
                    with col2:
                        agent2 = agents.get("agent2", {})
                        st.write(f"**Agent 2**: {agent2.get('name', '')}")
                    
                    st.divider()
                    
                    # 対話内容表示
                    for item in data.get("dialogue_history", []):
                        role = item.get("name", item.get("role", ""))
                        content = item.get("content", "")
                        phase = item.get("phase", "")
                        
                        avatar = "🎭" if "agent1" in item.get("role", "") else "🔬"
                        
                        with st.chat_message("assistant", avatar=avatar):
                            st.markdown(f"**{role}** ({phase})")
                            st.write(content)
                    
                except Exception as e:
                    st.error(f"読み込みエラー: {e}")
        else:
            st.info("保存済みの対話がありません")
    else:
        st.info("まだ対話が保存されていません")
        os.makedirs(dialogue_dir, exist_ok=True)

# ============ タブ4: 説明 ============
with tabs[3]:
    st.subheader("📖 システム説明")
    
    st.markdown("""
    ### 🎯 Advanced Dialogue System とは
    
    Director AI（監督AI）が2つのエージェントの対話を管理し、議論の質を向上させるシステムです。
    
    #### 主な特徴
    
    1. **Director による品質管理**
       - 毎ターン対話を分析
       - 最適な戦略を選択
       - 具体的な指示を生成
    
    2. **動的なキャラクター設定**
       - 6種類の個性的なキャラクター
       - 専門性と性格に基づく応答
       - 一貫した人格の維持
    
    3. **戦略的介入**
       - 議論の深化
       - 視点転換
       - 建設的対立
       - 収束と統合
    
    4. **フェーズ管理**
       - 探索 → 深化 → 収束 → 統合
       - 自動的なフェーズ移行
       - フェーズに応じた戦略選択
    
    #### 使い方
    
    1. サイドバーでキャラクターとモデルを選択
    2. テーマを入力または選択
    3. 「対話開始」ボタンをクリック
    4. 生成された対話を確認
    5. 必要に応じて保存
    
    #### キャラクター一覧
    
    - **哲学者ソクラテス**: 問いかけを通じて真理を探求
    - **科学者ダーウィン**: 観察と証拠に基づく論証
    - **創造的芸術家**: 直感と感性による独創的視点
    - **実践的エンジニア**: 問題解決と実装重視
    - **共感的カウンセラー**: 感情理解と対話促進
    - **分析的経済学者**: データと理論による分析
    
    #### カスタマイズ
    
    `config/` フォルダ内のJSONファイルを編集することで、
    キャラクター、戦略、プロンプトをカスタマイズできます。
    """)
    
    with st.expander("🔧 技術詳細"):
        st.markdown("""
        - **言語モデル**: Ollama (Qwen2.5, Gemma3等)
        - **アーキテクチャ**: 3層構造（UI / Manager / Core）
        - **設定管理**: JSON外部ファイル
        - **ストリーミング**: リアルタイム応答表示
        """)
    
    with st.expander("📝 更新履歴"):
        st.markdown("""
        - v1.0.0: 初期リリース
        - Director制御による対話品質管理
        - 6種類の基本キャラクター実装
        - 5つの介入戦略実装
        """)
