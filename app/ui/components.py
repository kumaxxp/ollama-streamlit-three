"""
UI Components for Advanced Dialogue System
StreamlitのUI部品を集約したモジュール
"""
import streamlit as st
from typing import Dict, Any, Optional, List
import json
from datetime import datetime


class DialogueUIComponents:
    """対話UI用の再利用可能なコンポーネント集"""
    
    @staticmethod
    def render_header(title: str = "🤖 Advanced Dialogue System") -> None:
        """ヘッダーの表示"""
        st.title(title)
        st.markdown("---")
    
    @staticmethod
    def render_theme_input() -> str:
        """テーマ入力UIの表示"""
        return st.text_input(
            "💭 議論のテーマを入力",
            placeholder="例: AIと人間の共存について",
            help="エージェントが議論するテーマを入力してください"
        )
    
    @staticmethod
    def render_agent_selector(agent_configs: Dict[str, Any]) -> tuple:
        """エージェント選択UIの表示"""
        col1, col2 = st.columns(2)
        
        with col1:
            agent1 = st.selectbox(
                "Agent 1",
                options=list(agent_configs.keys()),
                index=0
            )
        
        with col2:
            agent2 = st.selectbox(
                "Agent 2", 
                options=list(agent_configs.keys()),
                index=1
            )
        
        return agent1, agent2
    
    @staticmethod
    def render_control_buttons() -> Dict[str, bool]:
        """制御ボタンの表示"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            start = st.button("🎬 開始", type="primary", use_container_width=True)
        with col2:
            pause = st.button("⏸️ 一時停止", use_container_width=True)
        with col3:
            reset = st.button("🔄 リセット", use_container_width=True)
        
        return {
            "start": start,
            "pause": pause,
            "reset": reset
        }
    
    @staticmethod
    def render_agent_message(
        agent_name: str,
        message: str,
        avatar: str = "🤖",
        thinking_time: Optional[float] = None
    ) -> None:
        """エージェントメッセージの表示"""
        with st.chat_message(agent_name, avatar=avatar):
            st.write(message)
            if thinking_time:
                st.caption(f"⏱️ {thinking_time:.2f}秒")
    
    @staticmethod
    def render_director_intervention(
        analysis: Dict[str, Any],
        show_details: bool = False
    ) -> None:
        """Director介入の表示"""
        with st.expander("🎬 Director分析", expanded=show_details):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("深度", analysis.get("depth_level", 0))
            with col2:
                st.metric("関連性", f"{analysis.get('relevance', 0):.1f}")
            with col3:
                st.metric("建設性", f"{analysis.get('constructiveness', 0):.1f}")
            
            if analysis.get("intervention_needed"):
                st.warning(f"🎯 介入: {analysis.get('strategy', 'なし')}")
                st.info(f"📝 理由: {analysis.get('reason', '')}")
    
    @staticmethod
    def render_statistics(stats: Dict[str, Any]) -> None:
        """統計情報の表示"""
        with st.sidebar:
            st.subheader("📊 対話統計")
            st.metric("ターン数", stats.get("turn_count", 0))
            st.metric("総トークン数", stats.get("total_tokens", 0))
            st.metric("平均応答時間", f"{stats.get('avg_response_time', 0):.2f}秒")
            
            if stats.get("phase_progress"):
                st.progress(
                    stats["phase_progress"],
                    text=f"フェーズ: {stats.get('current_phase', '探索')}"
                )
    
    @staticmethod
    def render_export_section(dialogue_history: List[Dict]) -> None:
        """エクスポートセクションの表示"""
        with st.sidebar:
            st.subheader("💾 エクスポート")
            
            # JSON形式でダウンロード
            if st.button("📥 履歴をダウンロード"):
                json_str = json.dumps(
                    dialogue_history,
                    ensure_ascii=False,
                    indent=2
                )
                st.download_button(
                    label="JSON形式でダウンロード",
                    data=json_str,
                    file_name=f"dialogue_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )


class ParameterControls:
    """パラメータ制御UI"""
    
    @staticmethod
    def render_model_parameters() -> Dict[str, Any]:
        """モデルパラメータUIの表示"""
        with st.sidebar:
            st.subheader("⚙️ モデル設定")
            
            params = {}
            params["temperature"] = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=0.7,
                step=0.1
            )
            
            params["max_tokens"] = st.slider(
                "Max Tokens",
                min_value=50,
                max_value=500,
                value=150,
                step=50
            )
            
            params["top_p"] = st.slider(
                "Top P",
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                step=0.1
            )
            
            return params
    
    @staticmethod
    def render_director_settings() -> Dict[str, Any]:
        """Director設定UIの表示"""
        with st.sidebar:
            st.subheader("🎬 Director設定")
            
            settings = {}
            settings["intervention_threshold"] = st.slider(
                "介入閾値",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="この値以下の品質で介入"
            )
            
            settings["check_interval"] = st.number_input(
                "チェック間隔（ターン）",
                min_value=1,
                max_value=10,
                value=2,
                help="何ターンごとに品質チェックするか"
            )
            
            settings["auto_mode"] = st.checkbox(
                "自動モード",
                value=True,
                help="自動で対話を継続"
            )
            
            settings["show_analysis"] = st.checkbox(
                "分析を表示",
                value=False,
                help="Director分析を常に表示"
            )
            
            return settings


class DialogueDisplay:
    """対話表示用のヘルパークラス"""
    
    def __init__(self):
        self.container = st.container()
        self.messages = []
    
    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """メッセージを追加"""
        self.messages.append({
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        })
        
        # UIに表示
        with self.container:
            if role == "director":
                st.info(f"🎬 {content}")
            else:
                avatar = metadata.get("avatar", "🤖") if metadata else "🤖"
                with st.chat_message(role, avatar=avatar):
                    st.write(content)
    
    def clear(self) -> None:
        """表示をクリア"""
        self.messages = []
        self.container.empty()
    
    def get_history(self) -> List[Dict]:
        """履歴を取得"""
        return self.messages


class StreamingDisplay:
    """ストリーミング表示用のヘルパー"""
    
    def __init__(self, container):
        self.container = container
        self.placeholder = None
        self.content = ""
    
    def start(self, role: str, avatar: str = "🤖") -> None:
        """ストリーミング開始"""
        with self.container:
            with st.chat_message(role, avatar=avatar):
                self.placeholder = st.empty()
    
    def update(self, chunk: str) -> None:
        """コンテンツを更新"""
        self.content += chunk
        if self.placeholder:
            self.placeholder.markdown(self.content)
    
    def finish(self) -> str:
        """ストリーミング終了"""
        final_content = self.content
        self.content = ""
        self.placeholder = None
        return final_content