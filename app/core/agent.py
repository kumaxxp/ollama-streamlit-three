"""
エージェントクラス
各対話参加者の基本機能を提供
"""

import json
import ollama
from typing import Dict, List, Optional, Any
from datetime import datetime
import os

class Agent:
    """対話エージェントの基底クラス"""
    
    def __init__(
        self,
        agent_id: str,
        character_type: str,
        model_name: str = "qwen2.5:7b",
        temperature: float = 0.7
    ):
        """
        Args:
            agent_id: エージェントの識別子
            character_type: characters.jsonのキー
            model_name: 使用するOllamaモデル
            temperature: 生成時の温度パラメータ
        """
        self.agent_id = agent_id
        self.model_name = model_name
        self.temperature = temperature
        
        # キャラクター設定を読み込み
        self.character = self._load_character(character_type)
        self.character_type = character_type
        
        # プロンプトテンプレートを読み込み
        self.prompt_templates = self._load_prompt_templates()
        
        # 動的コンテキスト管理
        self.session_context = {}
        self.turn_directives = []
        self.memory = []
        self.response_count = 0
        
    def _load_character(self, character_type: str) -> Dict:
        """キャラクター設定を読み込み"""
        try:
            config_path = os.path.join("config", "characters.json")
            with open(config_path, "r", encoding="utf-8") as f:
                characters = json.load(f)
                if character_type not in characters:
                    raise ValueError(f"Character type '{character_type}' not found")
                return characters[character_type]
        except FileNotFoundError:
            print(f"Warning: characters.json not found at {config_path}")
            return self._get_default_character()
        except Exception as e:
            print(f"Error loading character: {e}")
            return self._get_default_character()
    
    def _get_default_character(self) -> Dict:
        """デフォルトキャラクター設定"""
        return {
            "name": "汎用エージェント",
            "personality": {
                "base": "中立的で建設的な議論参加者",
                "traits": ["論理的", "客観的", "協力的"],
                "communication_style": "明確で簡潔な説明"
            },
            "expertise": ["一般知識", "論理的思考"],
            "behavioral_rules": [
                "論理的に議論する",
                "相手の意見を尊重する",
                "建設的な提案を心がける"
            ],
            "speaking_patterns": {
                "opening": ["私の見解では〜", "〜と考えます"],
                "challenging": ["しかし〜はどうでしょうか", "別の観点から〜"],
                "agreeing": ["その通りです", "同意見です"]
            }
        }
    
    def _load_prompt_templates(self) -> Dict:
        """プロンプトテンプレートを読み込み"""
        try:
            config_path = os.path.join("config", "prompt_templates.json")
            with open(config_path, "r", encoding="utf-8") as f:
                templates = json.load(f)
                return templates["agent_prompt_template"]
        except FileNotFoundError:
            print(f"Warning: prompt_templates.json not found at {config_path}")
            return self._get_default_templates()
        except Exception as e:
            print(f"Error loading templates: {e}")
            return self._get_default_templates()
    
    def _get_default_templates(self) -> Dict:
        """デフォルトテンプレート"""
        return {
            "base": "あなたは{character_base}です。",
            "session": "テーマ: {theme}",
            "turn": "指示: {director_instruction}",
            "response": "相手の発言: {opponent_message}\n\n応答してください。"
        }
    
    def set_session_context(self, theme: str, goal: str = "建設的な議論", phase: str = "探索"):
        """セッションコンテキストを設定"""
        self.session_context = {
            "theme": theme,
            "discussion_goal": goal,
            "current_phase": phase
        }
    
    def add_directive(self, instruction: str, attention_points: List[str] = None):
        """Directorからの指示を追加"""
        directive = {
            "instruction": instruction,
            "attention_points": attention_points or [],
            "timestamp": datetime.now().isoformat()
        }
        self.turn_directives.append(directive)
        
        # 古い指示を削除（最新5つまで保持）
        if len(self.turn_directives) > 5:
            self.turn_directives = self.turn_directives[-5:]
    
    def add_to_memory(self, role: str, content: str):
        """メモリに発言を追加"""
        self.memory.append({
            "role": role,
            "content": content,
            "turn": len(self.memory)
        })
        
        # メモリサイズ制限（最新20発言まで）
        if len(self.memory) > 20:
            self.memory = self.memory[-20:]
    
    def build_prompt(self, opponent_message: str) -> str:
        """完全なプロンプトを構築"""
        # Base prompt (キャラクター設定)
        character_traits = ", ".join(self.character["personality"]["traits"])
        character_expertise = ", ".join(self.character["expertise"])
        behavioral_rules = "\n".join(f"- {rule}" for rule in self.character["behavioral_rules"])
        
        base_prompt = self.prompt_templates["base"].format(
            character_base=self.character["personality"]["base"],
            character_traits=character_traits,
            character_expertise=character_expertise,
            behavioral_rules=behavioral_rules
        )
        
        # Session prompt (セッション設定)
        session_prompt = self.prompt_templates["session"].format(
            theme=self.session_context.get("theme", "未設定"),
            discussion_goal=self.session_context.get("discussion_goal", "建設的な議論"),
            current_phase=self.session_context.get("current_phase", "探索")
        )
        
        # Turn prompt (ターン毎の指示)
        recent_directive = self.turn_directives[-1] if self.turn_directives else {
            "instruction": "自由に議論してください",
            "attention_points": []
        }
        recent_context = self._get_recent_context()
        attention_points = "\n".join(f"- {point}" for point in recent_directive["attention_points"])
        
        turn_prompt = self.prompt_templates["turn"].format(
            recent_context=recent_context,
            director_instruction=recent_directive["instruction"],
            attention_points=attention_points
        )
        
        # Response prompt (応答生成指示)
        response_prompt = self.prompt_templates["response"].format(
            opponent_message=opponent_message
        )
        
        return base_prompt + session_prompt + turn_prompt + response_prompt
    
    def _get_recent_context(self, max_turns: int = 3) -> str:
        """直近の文脈を取得"""
        if not self.memory:
            return "（議論開始）"
        
        recent = self.memory[-max_turns:]
        context_lines = []
        for item in recent:
            role = "自分" if item["role"] == self.agent_id else item["role"]
            # 長い文は省略
            content = item['content']
            if len(content) > 100:
                content = content[:100] + "..."
            context_lines.append(f"{role}: {content}")
            
        return "\n".join(context_lines)
    
    def generate_response(self, opponent_message: str, stream: bool = False) -> str:
        """応答を生成"""
        prompt = self.build_prompt(opponent_message)
        
        try:
            if stream:
                # ストリーミング対応（UIで使用）
                response_stream = ollama.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": opponent_message}
                    ],
                    options={
                        "temperature": self.temperature,
                        "num_predict": 200,
                        "stream": True
                    }
                )
                
                full_response = ""
                for chunk in response_stream:
                    if 'message' in chunk and 'content' in chunk['message']:
                        full_response += chunk['message']['content']
                        yield chunk['message']['content']
                
                # メモリに追加
                self.add_to_memory(self.agent_id, full_response)
                self.response_count += 1
                
            else:
                # 通常の応答
                response = ollama.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": opponent_message}
                    ],
                    options={
                        "temperature": self.temperature,
                        "num_predict": 200
                    }
                )
                
                generated_text = response['message']['content']
                
                # メモリに追加
                self.add_to_memory(self.agent_id, generated_text)
                self.response_count += 1
                
                return generated_text
                
        except Exception as e:
            error_msg = f"[応答生成エラー: {e}]"
            print(error_msg)
            return error_msg
    
    def get_character_info(self) -> Dict:
        """キャラクター情報を取得"""
        return {
            "id": self.agent_id,
            "name": self.character["name"],
            "type": self.character_type,
            "personality": self.character["personality"]["base"],
            "expertise": self.character["expertise"],
            "response_count": self.response_count
        }
    
    def get_speaking_pattern(self, pattern_type: str = "opening") -> str:
        """話し方パターンを取得"""
        patterns = self.character.get("speaking_patterns", {})
        pattern_list = patterns.get(pattern_type, [""])
        if pattern_list:
            import random
            return random.choice(pattern_list)
        return ""
    
    def reset(self):
        """エージェントをリセット"""
        self.session_context = {}
        self.turn_directives = []
        self.memory = []
        self.response_count = 0
        
    def get_state(self) -> Dict:
        """現在の状態を取得（デバッグ用）"""
        return {
            "agent_id": self.agent_id,
            "character_type": self.character_type,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "session_context": self.session_context,
            "directive_count": len(self.turn_directives),
            "memory_size": len(self.memory),
            "response_count": self.response_count
        }