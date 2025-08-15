"""
Streamlit Helper Functions
元 03_Advanced_Dialogue.py（現: 03_Advanced_Dialogue_Refactored.py）から抽出した共通処理（モデル名修正版）
"""
import streamlit as st
import ollama
_HAS_OLLAMA = True
from typing import List, Dict, Optional
from app.core.model_utils import ModelManager

@st.cache_data(ttl=300)
def get_available_models() -> List[str]:
    """
    Ollamaで利用可能なモデル一覧を取得（キャッシュ付き）
    実際にインストールされているモデル名を正確に取得
    
    Returns:
        利用可能なモデル名のリスト
    """
    mm = ModelManager()

    if not _HAS_OLLAMA:
        st.warning("⚠️ `ollama` ライブラリが見つかりません。ローカル実行時はollamaをインストールしてください。フォールバックのモデルリストを使用します。")
        return mm.get_fallback_models()

    try:
        available = mm.get_available_models()
        sorted_models = mm.get_sorted_models(available)

        if not sorted_models:
            st.warning("⚠️ Ollamaモデルが見つかりません。以下のコマンドでインストールしてください：")
            st.code("""
# 推奨モデルのインストール
ollama pull qwen2.5:7b
ollama pull gemma2:2b

# または量子化版（高速・省メモリ）
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull gemma3:4b
            """)
            return mm.get_fallback_models()

        return sorted_models

    except Exception as e:
        err_msg = str(e)
        if isinstance(e, (OSError, IOError)):
            st.error(f"⚠️ モデル取得時のI/Oエラー: {err_msg}")
            st.info("ローカルの ollama サーバやファイルアクセスに問題がある可能性があります。`ollama serve` が起動しているか、権限/ソケットを確認してください。")
        else:
            st.warning(f"⚠️ モデル取得エラー: {err_msg}")
        return mm.get_fallback_models()

def check_and_suggest_model(model_name: str) -> Optional[str]:
    """
    モデルが存在するか確認し、代替を提案
    
    Args:
        model_name: 確認するモデル名
    
    Returns:
        代替モデル名（存在する場合）、なければNone
    """
    available = get_available_models()
    
    if model_name in available:
        return model_name
    
    # 類似モデルを検索
    base_name = model_name.split(':')[0]
    for model in available:
        if base_name in model:
            return model
    
    # デフォルトを返す
    return available[0] if available else None

def render_model_selector(
    label: str,
    key: str,
    default_index: int = 0,
    help_text: Optional[str] = None
) -> str:
    """
    モデル選択UIを表示
    
    Args:
        label: 表示ラベル
        key: Streamlitのキー
        default_index: デフォルト選択インデックス
        help_text: ヘルプテキスト
    
    Returns:
        選択されたモデル名
    """
    models = get_available_models()
    
    if not models:
        st.error("モデルが見つかりません")
        return ""
    
    selected = st.selectbox(
        label,
        options=models,
        index=min(default_index, len(models)-1),
        key=key,
        help=help_text
    )
    
    # 推奨モデルの場合はバッジを表示
    if selected:
        if "qwen2.5:7b-instruct-q4_K_M" in selected:
            st.success("✅ 推奨モデル（日本語対話に最適・量子化版）")
        elif "qwen" in selected.lower():
            st.info("✓ Qwenモデル（日本語対応）")
        elif "gemma3:4b" in selected or "gemma2:2b" in selected:
            st.info("✓ Director推奨（高速判断）")
        elif "gemma" in selected.lower():
            st.info("✓ Gemmaモデル（バランス型）")
    
    return selected

def get_character_options() -> Dict[str, str]:
    """
    利用可能なキャラクター一覧をconfig/characters.jsonから取得
    Returns:
        キャラクターID -> 表示名の辞書
    """
    import os
    import json
    char_path_candidates = [
        'config/characters.json',
        './config/characters.json',
        '../config/characters.json',
        '../../config/characters.json',
        os.path.join(os.path.dirname(__file__), '../../config/characters.json')
    ]
    characters_data = None
    for path in char_path_candidates:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    characters_data = json.load(f)
                break
            except Exception:
                continue
    if not characters_data or 'characters' not in characters_data:
        # フォールバック
        return {}
    return {k: v.get('name', k) for k, v in characters_data['characters'].items()}

def get_theme_options() -> List[str]:
    """
    テーマオプションのリストを取得
    
    Returns:
        テーマのリスト
    """
    return [
        "AIと人間の共存について",
        "理想的な教育とは何か",
        "幸せな人生とは",
        "環境問題への取り組み",
        "これからの働き方",
        "SNSのメリットとデメリット",
        "お金と幸福の関係",
        "カスタム（下に入力）"
    ]

def display_dialogue_turn(
    turn_data: Dict,
    container: st.container
) -> None:
    """
    対話ターンを表示
    
    Args:
        turn_data: ターンデータ
        container: 表示先のコンテナ
    """
    with container:
        speaker = turn_data.get('speaker', '不明')
        listener = turn_data.get('listener', '相手')
        message = turn_data.get('message', '')
        turn_num = turn_data.get('turn', 0)
        
        if speaker == "Director":
            st.info(f"🎬 **Director** → {listener}")
            st.write(message)
        else:
            # キャラクターアイコンを設定
            icon = get_character_icon(speaker)
            st.markdown(f"**{icon} {speaker}** → {listener} (Turn {turn_num})")
            st.write(message)
        
        st.divider()

def get_character_icon(speaker_name: str) -> str:
    """
    キャラクター名からアイコンを取得
    
    Args:
        speaker_name: 話者名
    
    Returns:
        アイコン絵文字
    """
    icon_map = {
        "高校": "👧",
        "営業": "👔",
        "会社": "👔",
        "大学": "📚",
        "主婦": "👩",
        "フリー": "💻",
        "教師": "👨‍🏫",
        "AI": "🤖",
        "やな": "🎭",
        "あゆ": "🎨"
    }
    
    for key, icon in icon_map.items():
        if key in speaker_name:
            return icon
    
    return "👤"

def save_dialogue_json(dialogue_data: Dict) -> bytes:
    """
    対話データをJSON形式で保存用に変換
    
    Args:
        dialogue_data: 対話データ
    
    Returns:
        JSONバイトデータ
    """
    import json
    from datetime import datetime
    
    # タイムスタンプ追加
    dialogue_data["exported_at"] = datetime.now().isoformat()
    
    json_str = json.dumps(
        dialogue_data,
        ensure_ascii=False,
        indent=2
    )
    
    return json_str.encode('utf-8')

def check_ollama_connection() -> bool:
    """
    Ollama接続を確認
    
    Returns:
        接続可能ならTrue
    """
    if not _HAS_OLLAMA:
        # ollama ライブラリがそもそも無い場合は接続確認はスキップするが
        # ユーザーに警告を出してアプリを継続できるようにする。
        st.warning("⚠️ `ollama` ライブラリが見つかりません。ローカルに ollama をインストールしていない場合、実際のモデル呼び出しはできません。")
        return True
    try:
        client = ollama.Client()
        client.list()
        return True
    except Exception:
        return False

def show_connection_error() -> None:
    """Ollama接続エラーメッセージを表示"""
    st.error("❌ Ollamaに接続できません")
    st.info("""
    以下を確認してください：
    1. Ollamaが起動しているか: `ollama serve`
    2. モデルがインストールされているか: `ollama list`
    3. 推奨モデルをインストール:
    """)
    st.code("""
# 標準版
ollama pull qwen2.5:7b
ollama pull gemma2:2b

# または量子化版（推奨）
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull gemma3:4b
    """)

def create_download_button(
    data: bytes,
    filename: str,
    label: str = "📥 ダウンロード"
) -> None:
    """
    ダウンロードボタンを作成
    
    Args:
        data: ダウンロードデータ
        filename: ファイル名
        label: ボタンラベル
    """
    st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime="application/json"
    )