"""
動的対話生成システム - Streamlit版
語り手と批評家の対話を自動生成
"""

import streamlit as st
import ollama
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
import os

# ページ設定
st.set_page_config(
    page_title="動的対話生成",
    page_icon="🎭",
    layout="wide"
)

# タイトル
st.title("🎭 動的対話生成システム")
st.caption("テーマから語り手と批評家の対話を自動生成")

# サイドバー設定
with st.sidebar:
    st.header("⚙️ 設定")
    
    # モデル選択
    available_models = [m.model for m in ollama.list().models]
    
    narrator_model = st.selectbox(
        "語り手モデル",
        available_models,
        index=available_models.index("qwen2.5:7b") if "qwen2.5:7b" in available_models else 0
    )
    
    critic_model = st.selectbox(
        "批評家モデル", 
        available_models,
        index=available_models.index("gemma3:4b") if "gemma3:4b" in available_models else 0
    )
    
    # パラメータ設定
    with st.expander("詳細設定", expanded=False):
        max_turns = st.slider("最大ターン数", 4, 20, 8)
        narrator_temp = st.slider("語り手の創造性", 0.1, 1.0, 0.7, 0.1)
        critic_temp = st.slider("批評家の厳しさ", 0.1, 1.0, 0.6, 0.1)

# メインエリア
tab1, tab2, tab3 = st.tabs(["🎬 対話生成", "📊 分析", "📝 履歴"])

with tab1:
    # テーマ選択
    st.subheader("📌 テーマ選択")
    
    themes = [
        "火星コロニーで発見された謎の信号",
        "深夜のコンビニに現れた透明人間",
        "AIロボットが見た初めての夢",
        "江戸時代の寿司屋に現れたタイムトラベラー",
        "深海1万メートルの研究施設で起きた事件",
        "量子コンピュータの中に生まれた意識",
        "月面都市での殺人事件",
        "カスタム（自由入力）"
    ]
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_theme = st.selectbox("テーマを選択", themes)
        
        if selected_theme == "カスタム（自由入力）":
            custom_theme = st.text_input("テーマを入力してください")
            if custom_theme:
                selected_theme = custom_theme
    
    with col2:
        generate_btn = st.button("🎬 対話を生成", type="primary", use_container_width=True)
    
    # 対話生成エリア
    if generate_btn and selected_theme != "カスタム（自由入力）":
        # セッション状態初期化
        st.session_state.current_dialogue = []
        st.session_state.current_theme = selected_theme
        
        # プログレス表示
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 対話表示エリア
        dialogue_container = st.container()
        
        # 批評コンテキスト生成
        with st.spinner("🧠 批評設定を生成中..."):
            critic_context = generate_critic_context(selected_theme, critic_model)
            st.session_state.critic_context = critic_context
        
        # 対話生成ループ
        for turn in range(max_turns):
            progress = (turn + 1) / max_turns
            progress_bar.progress(progress)
            
            # 語り手の発言
            if turn == 0 or st.session_state.get("next_speaker") != "critic":
                status_text.text(f"語り手が話しています... (ターン {turn+1}/{max_turns})")
                narrator_text = generate_narrator_response(
                    selected_theme,
                    narrator_model,
                    narrator_temp,
                    turn,
                    st.session_state.current_dialogue
                )
                
                with dialogue_container:
                    with st.chat_message("assistant", avatar="🎭"):
                        st.write(f"**語り手**: {narrator_text}")
                
                st.session_state.current_dialogue.append({
                    "role": "narrator",
                    "content": narrator_text,
                    "turn": turn
                })
            
            # 批評家の発言
            if turn < max_turns - 1:
                status_text.text(f"批評家が考えています... (ターン {turn+1}/{max_turns})")
                critic_text = generate_critic_response(
                    narrator_text,
                    critic_model,
                    critic_temp,
                    critic_context,
                    turn
                )
                
                with dialogue_container:
                    with st.chat_message("user", avatar="🔍"):
                        st.write(f"**批評家**: {critic_text}")
                
                st.session_state.current_dialogue.append({
                    "role": "critic",
                    "content": critic_text,
                    "turn": turn
                })
        
        progress_bar.progress(1.0)
        status_text.text("✅ 対話生成完了！")
        
        # 保存ボタン
        if st.button("💾 対話を保存"):
            save_dialogue(st.session_state.current_dialogue, selected_theme)
            st.success("保存しました！")

with tab2:
    st.subheader("📊 対話分析")
    
    if "current_dialogue" in st.session_state and st.session_state.current_dialogue:
        analysis = analyze_dialogue(st.session_state.current_dialogue)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("総ターン数", analysis["total_turns"])
        with col2:
            st.metric("矛盾指摘", analysis["contradiction_count"])
        with col3:
            st.metric("平均文字数", f"{analysis['avg_length']:.1f}")
        
        # パターン分析
        if analysis["patterns"]:
            st.subheader("批評パターン")
            st.bar_chart(analysis["patterns"])
        
        # 批評コンテキスト表示
        if "critic_context" in st.session_state:
            with st.expander("批評設定詳細"):
                st.json(st.session_state.critic_context)
    else:
        st.info("対話を生成すると分析結果が表示されます")

with tab3:
    st.subheader("📝 保存済み対話")
    
    # 履歴ファイル一覧
    if os.path.exists("data/dialogues"):
        files = sorted([f for f in os.listdir("data/dialogues") if f.endswith(".json")], reverse=True)
        
        if files:
            selected_file = st.selectbox("履歴を選択", files)
            
            if st.button("📂 読み込み"):
                with open(f"data/dialogues/{selected_file}", "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                st.write(f"**テーマ**: {data['theme']}")
                st.write(f"**生成日時**: {data['timestamp']}")
                
                for item in data['dialogue']:
                    role = "🎭 語り手" if item['role'] == 'narrator' else "🔍 批評家"
                    st.write(f"{role}: {item['content']}")
        else:
            st.info("保存済みの対話がありません")

# ============ ヘルパー関数 ============

def generate_critic_context(theme: str, model: str) -> Dict[str, Any]:
    """批評用コンテキストを生成"""
    prompt = f"""
テーマ「{theme}」の物語を批評するための設定を生成してください。

以下のJSON形式で出力（説明不要）:
{{
  "facts": ["重要な事実1", "重要な事実2", "重要な事実3"],
  "contradictions": ["よくある矛盾1", "よくある矛盾2"],
  "personality": "批評者の性格",
  "focus": ["注目点1", "注目点2"],
  "forbidden": ["存在しないもの1", "存在しないもの2"]
}}
"""
    
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.3, "num_predict": 500}
    )
    
    content = response['message']['content']
    
    # JSON抽出
    try:
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except:
        pass
    
    # フォールバック
    return {
        "facts": ["物理法則に従う", "論理的整合性が必要", "因果関係が明確"],
        "contradictions": ["前後の矛盾", "設定の無視"],
        "personality": "懐疑的",
        "focus": ["一貫性", "論理性"],
        "forbidden": ["矛盾", "非論理的展開"]
    }

def generate_narrator_response(theme: str, model: str, temperature: float, turn: int, dialogue: List) -> str:
    """語り手の応答生成"""
    
    if turn == 0:
        prompt = f"「{theme}」の物語を始めてください。具体的な場面から2文で。"
    else:
        last_critic = dialogue[-1]['content'] if dialogue and dialogue[-1]['role'] == 'critic' else ""
        prompt = f"批評「{last_critic}」を受けて、物語を続けてください。2文で。"
    
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": f"あなたは「{theme}」の物語を語る語り手です。"},
            {"role": "user", "content": prompt}
        ],
        options={"temperature": temperature, "num_predict": 100}
    )
    
    text = response['message']['content']
    
    # 2文に制限
    sentences = re.split(r'[。！？]', text)
    sentences = [s for s in sentences if s.strip()][:2]
    return '。'.join(sentences) + '。'

def generate_critic_response(narrator_text: str, model: str, temperature: float, context: Dict, turn: int) -> str:
    """批評家の応答生成"""
    
    system_prompt = f"""
あなたは{context.get('personality', '懐疑的')}な批評家です。
重要な事実: {', '.join(context.get('facts', [])[:2])}
存在してはいけないもの: {', '.join(context.get('forbidden', []))}
返答は15文字以内で簡潔に。
"""
    
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"語り手: {narrator_text}\n\n短く反応してください。"}
        ],
        options={"temperature": temperature, "num_predict": 40}
    )
    
    text = response['message']['content']
    
    # 長さ制限
    if len(text) > 20:
        text = text[:20]
    
    return text

def analyze_dialogue(dialogue: List[Dict]) -> Dict[str, Any]:
    """対話の分析"""
    analysis = {
        "total_turns": len(dialogue),
        "contradiction_count": 0,
        "patterns": {},
        "avg_length": 0
    }
    
    lengths = []
    for item in dialogue:
        lengths.append(len(item['content']))
        
        # 批評パターン分析
        if item['role'] == 'critic':
            if "ない" in item['content'] or "おかしい" in item['content']:
                analysis["contradiction_count"] += 1
                analysis["patterns"]["矛盾指摘"] = analysis["patterns"].get("矛盾指摘", 0) + 1
            elif "？" in item['content']:
                analysis["patterns"]["質問"] = analysis["patterns"].get("質問", 0) + 1
            else:
                analysis["patterns"]["相槌"] = analysis["patterns"].get("相槌", 0) + 1
    
    if lengths:
        analysis["avg_length"] = sum(lengths) / len(lengths)
    
    return analysis

def save_dialogue(dialogue: List[Dict], theme: str):
    """対話を保存"""
    os.makedirs("data/dialogues", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/dialogues/dialogue_{timestamp}.json"
    
    data = {
        "theme": theme,
        "dialogue": dialogue,
        "timestamp": timestamp,
        "analysis": analyze_dialogue(dialogue)
    }
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)