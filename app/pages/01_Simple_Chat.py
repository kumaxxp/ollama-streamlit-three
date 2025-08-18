"""
Streamlit + Ollama シンプルチャットアプリ
Qwen2.5-7bモデルとの対話インターフェース
"""

import streamlit as st
import ollama
from datetime import datetime
import json
import re

# ページ設定
st.set_page_config(
    page_title="Ollama Chat - Qwen2.5",
    page_icon="🤖",
    layout="wide"
)

# セッション状態の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model" not in st.session_state:
    st.session_state.model = "qwen2.5:7b"
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7

# サイドバー設定
with st.sidebar:
    st.title("⚙️ 設定")
    
    # モデル選択
    try:
        # 利用可能なモデルを取得
        models = ollama.list()
        model_names = [model['name'] for model in models['models']]
        
        # Qwen2.5-7bがあるか確認
        if "qwen2.5:7b" not in model_names and "qwen2.5:latest" not in model_names:
            st.warning("⚠️ Qwen2.5-7bが見つかりません")
            st.code("ollama pull qwen2.5:7b", language="bash")
            # 代替モデルを選択
            if model_names:
                st.session_state.model = st.selectbox(
                    "代替モデル選択",
                    model_names,
                    index=0
                )
        else:
            st.session_state.model = st.selectbox(
                "モデル選択",
                model_names,
                index=model_names.index("qwen2.5:7b") if "qwen2.5:7b" in model_names else 0
            )
    except Exception as e:
        st.error(f"Ollama接続エラー: {e}")
        st.info("Ollamaが起動していることを確認してください")
        st.code("sudo systemctl status ollama", language="bash")
    
    st.divider()
    
    # パラメータ設定
    st.session_state.temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=st.session_state.temperature,
        step=0.1,
        help="高いほど創造的、低いほど一貫性のある応答"
    )
    
    top_p = st.slider(
        "Top P",
        min_value=0.0,
        max_value=1.0,
        value=0.9,
        step=0.05,
        help="トークン選択の累積確率閾値"
    )
    
    max_tokens = st.number_input(
        "最大トークン数",
        min_value=50,
        max_value=4096,
        value=1024,
        step=50
    )
    
    st.divider()
    
    # 会話履歴管理
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ 履歴クリア", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        # 履歴をJSONで保存
        if st.button("💾 履歴保存", use_container_width=True):
            if st.session_state.messages:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"chat_history_{timestamp}.json"
                json_str = json.dumps(st.session_state.messages, ensure_ascii=False, indent=2)
                st.download_button(
                    label="📥 ダウンロード",
                    data=json_str,
                    file_name=filename,
                    mime="application/json"
                )
    
    st.divider()
    
    # システム情報
    with st.expander("📊 システム情報"):
        st.code(f"""
モデル: {st.session_state.model}
Temperature: {st.session_state.temperature}
Top P: {top_p}
最大トークン: {max_tokens}
メッセージ数: {len(st.session_state.messages)}
        """)

# メインチャット画面
st.title("🤖 Ollama Chat Interface")
st.caption(f"使用モデル: {st.session_state.model}")


def _sanitize_output(text: str) -> str:
    """UI表示用に思考過程（CoT）を除去する軽量フィルタ"""
    try:
        s = str(text)
        # タグ形式
        s = re.sub(r"(?is)<\s*(think|thought|scratchpad)\b[^>]*>.*?<\s*/\s*\1\s*>", "", s)
        s = re.sub(r"(?is)<\s*(think|thought|scratchpad)\b[^>]*>[\s\S]*$", "", s)
        # フェンス付きコードブロック
        s = re.sub(r"(?is)```\s*(reasoning|thought|think)[\s\S]*?```", "", s)
        s = re.sub(r"(?is)```\s*(reasoning|thought|think)[\s\S]*$", "", s)
        # 日本語ラベル
        s = re.sub(r"(?s)【思考】[\s\S]*?【/思考】", "", s)
        s = re.sub(r"(?s)【思考】[\s\S]*$", "", s)
        # 行頭の『思考:』など
        s = re.sub(r"(?mi)^(思考|推論|考え|Reasoning|Thoughts?)\s*[:：].*$\n?", "", s)
        # 連続改行の整理
        s = re.sub(r"\n{3,}", "\n\n", s).strip()
        return s
    except Exception:
        return text

# チャット履歴表示
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 入力フィールド
if prompt := st.chat_input("メッセージを入力してください..."):
    # ユーザーメッセージを追加
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # ユーザーメッセージを表示
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # アシスタントの応答
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # ストリーミング応答
            with st.spinner("考え中..."):
                # Ollamaにリクエスト送信
                stream = ollama.chat(
                    model=st.session_state.model,
                    messages=st.session_state.messages,
                    stream=True,
                    options={
                        "temperature": st.session_state.temperature,
                        "top_p": top_p,
                        # 安全な既定: KVキャッシュ/VRAM圧迫を抑える
                        "num_ctx": 4096,
                        "num_batch": 128,
                        "num_predict": max_tokens,
                    }
                )
                
                # ストリーミング表示
                for chunk in stream:
                    if chunk['message']['content']:
                        full_response += chunk['message']['content']
                        display_text = _sanitize_output(full_response)
                        message_placeholder.markdown(display_text + "▌")
                
                message_placeholder.markdown(_sanitize_output(full_response))
        
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
            full_response = "申し訳ございません。応答の生成中にエラーが発生しました。"
            message_placeholder.markdown(full_response)
    
    # アシスタントメッセージを履歴に追加
    st.session_state.messages.append({"role": "assistant", "content": _sanitize_output(full_response)})

# フッター
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🚀 Powered by Ollama")
with col2:
    st.caption("🎯 RTX A5000 Optimized")
with col3:
    st.caption("📝 Ubuntu 24.04")
