"""
Advanced Dialogue System - Streamlit UI
自発的なDirector制御と一般人キャラクターによる自然な対話
動的なモデル選択機能付き
"""

import streamlit as st
import json
import asyncio
from datetime import datetime
import sys
import os
from pathlib import Path
import ollama

# パスの設定
sys.path.append(str(Path(__file__).parent.parent.parent))

# 必要なモジュールのインポート
from app.core.dialogue_manager import DialogueManager
from app.core.agent import Agent
from app.core.director import AutonomousDirector

# ページ設定
st.set_page_config(
    page_title="Advanced Dialogue System",
    page_icon="🎭",
    layout="wide"
)

# タイトルと説明
st.title("🎭 Advanced Dialogue System")
st.markdown("""
**自然な対話生成システム**  
一般的なキャラクター（高校生、会社員、主婦など）による議論を、
Director AIが自発的に監督・改善します。
""")

# セッション状態の初期化
if 'dialogue_manager' not in st.session_state:
    st.session_state.dialogue_manager = None
if 'dialogue_history' not in st.session_state:
    st.session_state.dialogue_history = []
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'current_turn' not in st.session_state:
    st.session_state.current_turn = 0

# サイドバー設定
with st.sidebar:
    st.header("⚙️ 対話設定")
    
    # テーマ選択
    st.subheader("📝 議論テーマ")
    theme_options = [
        "AIと人間の共存について",
        "理想的な教育とは何か",
        "幸せな人生とは",
        "環境問題への取り組み",
        "これからの働き方",
        "SNSのメリットとデメリット",
        "お金と幸福の関係",
        "カスタム（下に入力）"
    ]
    selected_theme = st.selectbox("テーマを選択", theme_options)
    
    if selected_theme == "カスタム（下に入力）":
        custom_theme = st.text_input("カスタムテーマ", placeholder="議論したいテーマを入力")
        theme = custom_theme if custom_theme else "自由討論"
    else:
        theme = selected_theme
    
    st.divider()
    
    # キャラクター選択
    st.subheader("👥 キャラクター選択")
    
    # キャラクターオプション
    character_options = {
        "AI-tuber-college_student_girl": "やな（女子大生っぽいAI・明るい・うかつ）",
        "AI-tuber-high_school_girl": "あゆ（女子高生っぽいAI・冷静沈着）",
        "high_school_girl_optimistic": "さくら（高校2年生・明るい）",
        "office_worker_tired": "田中（32歳・営業職）",
        "college_student_curious": "ユウキ（大学3年生・哲学科）",
        "housewife_practical": "美咲（28歳・主婦）",
        "freelancer_creative": "レン（27歳・フリーランス）",
        "retired_wise": "山田（65歳・元教師）"
    }
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**キャラクター1**")
        char1_key = st.selectbox(
            "選択",
            list(character_options.keys()),
            format_func=lambda x: character_options[x],
            key="char1_select"
        )
    
    with col2:
        st.markdown("**キャラクター2**")
        # char1と異なるデフォルトを設定
        char2_options = [k for k in character_options.keys() if k != char1_key]
        if not char2_options:
            char2_options = list(character_options.keys())
        
        char2_key = st.selectbox(
            "選択",
            char2_options,
            format_func=lambda x: character_options[x],
            key="char2_select"
        )
    
    # キャラクター詳細表示
    if st.checkbox("キャラクター詳細を表示"):
        with st.expander("キャラクター1の詳細"):
            try:
                with open('config/characters.json', 'r', encoding='utf-8') as f:
                    characters = json.load(f)
                char1_info = characters['characters'].get(char1_key, {})
                st.json(char1_info)
            except:
                st.error("キャラクター情報を読み込めません")
        
        with st.expander("キャラクター2の詳細"):
            try:
                with open('config/characters.json', 'r', encoding='utf-8') as f:
                    characters = json.load(f)
                char2_info = characters['characters'].get(char2_key, {})
                st.json(char2_info)
            except:
                st.error("キャラクター情報を読み込めません")
    
    st.divider()
    
    # 詳細設定
    st.subheader("🎛️ 詳細設定")
    
    with st.expander("パラメータ設定", expanded=False):
        max_turns = st.slider("最大ターン数", 5, 30, 20)
        
        st.markdown("**モデル設定**")
        
        # Ollamaから利用可能なモデルを動的に取得
        @st.cache_data(ttl=300)  # 5分間キャッシュ
        def get_available_models():
            """Ollamaで利用可能なモデル一覧を取得"""
            try:
                client = ollama.Client()
                models_list = client.list()
                
                # モデル名を抽出
                model_names = []
                for model in models_list.get('models', []):
                    model_name = model.get('name', '')
                    if model_name:
                        model_names.append(model_name)
                
                # 推奨モデル（具体的な量子化版を優先）
                priority_models = [
                    # 最優先：本番環境推奨モデル
                    "qwen2.5:7b-instruct-q4_K_M",
                    "gemma3:12b",
                    "gpt-oss:20b",
                    "gemma3:4b",
                    "qwen:7b",
                    
                    # その他の推奨モデル
                    "qwen2.5:7b-instruct-q5_K_M",
                    "qwen2.5:7b-instruct",
                    "qwen2.5:14b-instruct-q4_K_M",
                    "gemma2:9b",
                    "llama3.2:3b",
                    "llama3.1:8b"
                ]
                
                # 優先モデルで利用可能なものを先頭に
                available_priority = []
                for model in priority_models:
                    if model in model_names:
                        available_priority.append(model)
                
                # その他のモデル
                other_models = sorted([m for m in model_names if m not in priority_models])
                
                final_list = available_priority + other_models
                
                if not final_list:
                    # モデルが見つからない場合のデフォルト
                    final_list = [
                        "qwen2.5:7b-instruct-q4_K_M",
                        "gemma3:12b",
                        "gemma3:4b"
                    ]
                    st.warning("⚠️ Ollamaモデルを取得できませんでした。推奨モデルをインストールしてください。")
                
                return final_list
                
            except Exception as e:
                st.warning(f"⚠️ モデル一覧の取得に失敗: {str(e)}")
                # エラー時のフォールバック（本番推奨モデル）
                return [
                    "qwen2.5:7b-instruct-q4_K_M",
                    "gemma3:12b",
                    "gpt-oss:20b",
                    "gemma3:4b",
                    "qwen:7b"
                ]
        
        # モデル一覧を取得
        model_options = get_available_models()
        
        # モデル選択UI
        col_model1, col_model2 = st.columns(2)
        
        with col_model1:
            # エージェント用モデル選択
            agent_model = st.selectbox(
                "エージェントモデル",
                model_options,
                index=0,
                help="対話エージェントが使用するモデル（推奨: qwen2.5:7b-instruct-q4_K_M）"
            )
            
            # モデル推奨情報
            if agent_model == "qwen2.5:7b-instruct-q4_K_M":
                st.success("✅ 推奨モデル（日本語対話に最適）")
            elif agent_model in ["gemma3:12b", "gpt-oss:20b"]:
                st.info("✓ 高品質モデル")
            
            # カスタムモデル入力オプション
            use_custom_agent = st.checkbox("カスタムモデル名を入力（エージェント）", key="custom_agent")
            if use_custom_agent:
                agent_model = st.text_input(
                    "モデル名を入力",
                    placeholder="例: qwen2.5:7b-instruct-q4_K_M",
                    key="custom_agent_input"
                ) or agent_model
        
        with col_model2:
            # Director用モデル選択
            director_model = st.selectbox(
                "Directorモデル",
                model_options,
                index=min(3, len(model_options)-1),  # gemma3:4bを優先
                help="監督AIが使用するモデル（推奨: gemma3:4b - 軽量で高速）"
            )
            
            # モデル推奨情報
            if director_model == "gemma3:4b":
                st.success("✅ Director推奨モデル（高速判断）")
            elif director_model == "qwen2.5:7b-instruct-q4_K_M":
                st.info("✓ 代替推奨モデル")
            
            # カスタムモデル入力オプション
            use_custom_director = st.checkbox("カスタムモデル名を入力（Director）", key="custom_director")
            if use_custom_director:
                director_model = st.text_input(
                    "モデル名を入力",
                    placeholder="例: gemma3:4b",
                    key="custom_director_input"
                ) or director_model
        
        # モデル情報表示
        if st.checkbox("モデル情報を表示"):
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.info(f"**エージェント**: {agent_model}")
                try:
                    client = ollama.Client()
                    info = client.show(agent_model)
                    st.json({
                        "parameters": info.get("details", {}).get("parameter_size", "不明"),
                        "quantization": info.get("details", {}).get("quantization_level", "不明")
                    })
                except:
                    st.text("詳細情報を取得できません")
            
            with col_info2:
                st.info(f"**Director**: {director_model}")
                try:
                    client = ollama.Client()
                    info = client.show(director_model)
                    st.json({
                        "parameters": info.get("details", {}).get("parameter_size", "不明"),
                        "quantization": info.get("details", {}).get("quantization_level", "不明")
                    })
                except:
                    st.text("詳細情報を取得できません")
        
        st.markdown("**温度設定**")
        col_temp1, col_temp2 = st.columns(2)
        with col_temp1:
            agent_temp = st.slider(
                "エージェント温度",
                0.1, 1.0, 0.7, 0.1,
                help="高いほど創造的、低いほど一貫性重視"
            )
        with col_temp2:
            director_temp = st.slider(
                "Director温度",
                0.1, 1.0, 0.3, 0.1,
                help="低めを推奨（判断の一貫性のため）"
            )
        
        enable_director = st.checkbox("Director介入を有効化", value=True)
        
        # モデル再読み込みボタン
        if st.button("🔄 モデル一覧を更新"):
            st.cache_data.clear()
            st.success("モデル一覧を更新しました")
            st.rerun()
    
    st.divider()
    
    # Director統計
    if st.session_state.dialogue_manager:
        st.subheader("📊 Director統計")
        stats = st.session_state.dialogue_manager.director.get_intervention_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("総介入回数", stats.get('total', 0))
        with col2:
            st.metric("現在ターン", st.session_state.current_turn)
        
        if stats.get('by_type'):
            st.markdown("**介入タイプ別**")
            for itype, count in stats['by_type'].items():
                st.text(f"{itype}: {count}回")

# メインエリア
col_main, col_stats = st.columns([3, 1])

with col_main:
    # 対話表示エリア
    dialogue_container = st.container()
    
    # コントロールボタン
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("🎬 対話を開始", disabled=st.session_state.is_running):
            # DialogueManagerの初期化
            client = ollama.Client()
            st.session_state.dialogue_manager = DialogueManager(client, director_model)
            
            # エージェント設定
            agent1_config = {
                'character_type': char1_key,
                'model': agent_model,
                'temperature': agent_temp
            }
            agent2_config = {
                'character_type': char2_key,
                'model': agent_model,
                'temperature': agent_temp
            }
            
            # 初期化
            st.session_state.dialogue_manager.initialize(
                theme, 
                agent1_config, 
                agent2_config
            )
            st.session_state.dialogue_manager.enable_director = enable_director
            st.session_state.dialogue_manager.max_turns = max_turns
            
            st.session_state.dialogue_history = []
            st.session_state.is_running = True
            st.session_state.current_turn = 0
            
            st.success("対話を開始しました！")
            st.rerun()
    
    with col_btn2:
        if st.button("⏸️ 一時停止", disabled=not st.session_state.is_running):
            st.session_state.is_running = False
            if st.session_state.dialogue_manager:
                st.session_state.dialogue_manager.stop_dialogue()
            st.info("対話を一時停止しました")
    
    with col_btn3:
        if st.button("🔄 リセット"):
            st.session_state.dialogue_manager = None
            st.session_state.dialogue_history = []
            st.session_state.is_running = False
            st.session_state.current_turn = 0
            st.info("リセットしました")
            st.rerun()

# 対話実行
async def run_dialogue_async():
    """非同期で対話を実行"""
    manager = st.session_state.dialogue_manager
    
    while st.session_state.is_running and st.session_state.current_turn < max_turns:
        try:
            # 1ターン実行
            turn_result = await manager.run_turn()
            st.session_state.current_turn += 1
            st.session_state.dialogue_history.append(turn_result)
            
            # 表示更新
            display_dialogue_turn(turn_result)
            
            # Director介入があれば表示
            if 'director_intervention' in turn_result:
                intervention = turn_result['director_intervention']
                st.info(f"🎬 **Director介入**: {intervention['reason']}")
                st.write(f"_{intervention['message']}_")
            
            # 少し待機
            await asyncio.sleep(1)
            
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
            st.session_state.is_running = False
            break
    
    st.session_state.is_running = False
    st.success("対話が完了しました！")

def display_dialogue_turn(turn_data):
    """対話ターンを表示"""
    with dialogue_container:
        speaker = turn_data.get('speaker', '不明')
        listener = turn_data.get('listener', '相手')
        message = turn_data.get('message', '')
        turn_num = turn_data.get('turn', 0)
        
        # スピーカーによって色を変える
        if speaker == "Director":
            st.info(f"🎬 **Director** → {listener}")
            st.write(message)
        else:
            # キャラクターアイコンを設定
            icon = "👤"
            if "高校" in speaker:
                icon = "👧"
            elif "営業" in speaker or "会社" in speaker:
                icon = "👔"
            elif "大学" in speaker:
                icon = "📚"
            elif "主婦" in speaker:
                icon = "👩"
            elif "フリー" in speaker:
                icon = "💻"
            elif "教師" in speaker or "先生" in speaker:
                icon = "👨‍🏫"
            
            st.markdown(f"**{icon} {speaker}** → {listener} (Turn {turn_num})")
            st.write(message)
        
        st.divider()

# 実行処理
if st.session_state.is_running and st.session_state.dialogue_manager:
    # 非同期実行
    asyncio.run(run_dialogue_async())

# 対話履歴の表示
with col_stats:
    st.subheader("📜 対話サマリー")
    
    if st.session_state.dialogue_manager:
        summary = st.session_state.dialogue_manager.get_summary()
        
        st.metric("テーマ", theme[:20] + "...")
        st.metric("総ターン数", summary.get('total_turns', 0))
        st.metric("Director介入", summary.get('director_interventions', 0))
        
        if summary.get('participants'):
            st.markdown("**参加者**")
            for p in summary['participants']:
                st.text(f"• {p}")

# 対話の保存
if st.session_state.dialogue_history and len(st.session_state.dialogue_history) > 0:
    st.divider()
    
    col_save1, col_save2 = st.columns(2)
    
    with col_save1:
        if st.button("💾 対話を保存"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dialogue_{timestamp}.json"
            filepath = os.path.join("data", "dialogues", filename)
            
            # ディレクトリ作成
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 保存
            if st.session_state.dialogue_manager:
                st.session_state.dialogue_manager.save_dialogue(filepath)
                st.success(f"対話を保存しました: {filename}")
    
    with col_save2:
        # ダウンロードボタン
        if st.session_state.dialogue_manager:
            save_data = {
                "summary": st.session_state.dialogue_manager.get_summary(),
                "dialogue": st.session_state.dialogue_history,
                "director_stats": st.session_state.dialogue_manager.director.get_intervention_stats()
            }
            
            json_str = json.dumps(save_data, ensure_ascii=False, indent=2)
            st.download_button(
                label="📥 JSONをダウンロード",
                data=json_str,
                file_name=f"dialogue_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# フッター
st.divider()
st.markdown("""
---
**Advanced Dialogue System v2.0**  
Director AIによる自発的な介入と一般人キャラクターによる自然な対話生成
""")