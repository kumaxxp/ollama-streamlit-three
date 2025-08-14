"""
Agent - 対話エージェントの実装
キャラクター設定に基づく自然な対話生成
"""

import json
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class Agent:
    """
    対話エージェント
    一般的なキャラクター（女子高生、会社員など）として振る舞う
    """
    
    def __init__(
        self,
        agent_id: str,
        character_type: str,
        model_name: str = "qwen2.5:7b",
        temperature: float = 0.7,
        ollama_client = None
    ):
        self.agent_id = agent_id
        self.character_type = character_type
        self.model_name = model_name
        self.temperature = temperature
        self.client = ollama_client
        
        # キャラクター設定を読み込み
        self.character = self._load_character(character_type)
        
        # セッション情報
        self.session_context = {}
        self.turn_directives = []
        self.memory = []  # 発言履歴
        
        logger.info(f"Agent {agent_id} initialized as {self.character['name']}")
    
    def _load_character(self, character_type: str) -> Dict:
        """キャラクター設定を読み込み"""
        try:
            with open('config/characters.json', 'r', encoding='utf-8') as f:
                characters = json.load(f)
                
            if character_type in characters['characters']:
                return characters['characters'][character_type]
            else:
                logger.warning(f"Character type {character_type} not found, using default")
                return self._get_default_character()
                
        except FileNotFoundError:
            logger.error("characters.json not found, using default character")
            return self._get_default_character()
        except Exception as e:
            logger.error(f"Error loading character: {e}")
            return self._get_default_character()
    
    def _get_default_character(self) -> Dict:
        """デフォルトキャラクター（フォールバック用）"""
        return {
            "name": "対話参加者",
            "personality": "中立的で建設的な議論を心がける",
            "speaking_style": "丁寧で分かりやすい",
            "background": "一般的な知識を持つ対話者",
            "values": "相互理解と建設的な議論",
            "behavioral_patterns": {
                "greeting": ["こんにちは", "よろしくお願いします"],
                "agreement": ["なるほど", "その通りですね"],
                "disagreement": ["別の見方もあるかもしれません", "私はこう思います"]
            }
        }
    
    def set_session_context(self, theme: str, goal: str, phase: str):
        """セッションコンテキストを設定"""
        self.session_context = {
            "theme": theme,
            "goal": goal,
            "phase": phase,
            "start_time": datetime.now().isoformat()
        }
    
    def add_directive(self, instruction: str, attention_points: List[str]):
        """Directorからの指示を追加"""
        self.turn_directives.append({
            "instruction": instruction,
            "attention_points": attention_points,
            "timestamp": datetime.now().isoformat()
        })
    
    def _build_system_prompt(self) -> str:
        """システムプロンプトを構築"""
        # キャラクターのプロンプトテンプレートを使用
        if 'prompt_template' in self.character:
            # テンプレートに値を埋め込み
            template = self.character['prompt_template']
            return template.format(
                name=self.character['name'],
                personality=self.character.get('personality', ''),
                speaking_style=self.character.get('speaking_style', ''),
                background=self.character.get('background', ''),
                values=self.character.get('values', '')
            )
        
        # テンプレートがない場合は標準形式
        return f"""
あなたは{self.character['name']}です。

【基本設定】
性格: {self.character.get('personality', '未設定')}
話し方: {self.character.get('speaking_style', '自然な日本語')}
背景: {self.character.get('background', '一般的な背景')}
価値観: {self.character.get('values', '建設的な対話')}

【重要な指示】
- あなたのキャラクターを一貫して保ってください
- 相手の発言をよく聞き、それに応答してください
- 自然な対話を心がけてください
- 議論を建設的に進めてください
"""
    
    def build_prompt(self, context: Dict) -> str:
        """
        対話用プロンプトを構築
        
        Args:
            context: {
                "opponent_name": str,
                "opponent_message": str,
                "recent_history": List[Dict],
                "director_instruction": Optional[str]
            }
        """
        opponent_name = context.get('opponent_name', '相手')
        opponent_message = context.get('opponent_message', '')
        recent_history = context.get('recent_history', [])
        director_instruction = context.get('director_instruction', '')
        
        # 基本プロンプト
        prompt_parts = [
            f"あなたは{self.character['name']}として、{opponent_name}と対話しています。",
            f"\nテーマ: {self.session_context.get('theme', '自由討論')}",
            f"現在のフェーズ: {self.session_context.get('phase', 'exploration')}\n"
        ]
        
        # 最近の履歴があれば追加
        if recent_history:
            prompt_parts.append("\n【これまでの対話】")
            for entry in recent_history[-3:]:  # 最新3つ
                speaker = entry.get('speaker', '不明')
                message = entry.get('message', '')
                prompt_parts.append(f"{speaker}: 「{message}」")
        
        # 相手の最新メッセージ
        if opponent_message:
            prompt_parts.append(f"\n【{opponent_name}の発言】")
            prompt_parts.append(f"「{opponent_message}」")
        
        # Director指示があれば追加
        if director_instruction:
            prompt_parts.append(f"\n【アドバイス】")
            prompt_parts.append(director_instruction)
        
        # 最新のDirective
        if self.turn_directives:
            latest_directive = self.turn_directives[-1]
            prompt_parts.append(f"\n【今回の注意点】")
            prompt_parts.append(latest_directive['instruction'])
            if latest_directive['attention_points']:
                prompt_parts.append("- " + "\n- ".join(latest_directive['attention_points']))
        
        # 応答指示
        prompt_parts.append(f"\n上記を踏まえて、{opponent_name}に対して{self.character['name']}として応答してください。")
        prompt_parts.append("自然で、あなたらしい発言を心がけてください。")
        
        return "\n".join(prompt_parts)
    
    async def generate_response(self, context: Dict) -> str:
        """
        応答を生成
        
        Args:
            context: 対話コンテキスト
        
        Returns:
            生成された応答
        """
        prompt = self.build_prompt(context)
        system_prompt = self._build_system_prompt()
        
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                options={
                    "temperature": self.temperature,
                    "top_p": 0.95,
                    "seed": None
                },
                stream=False
            )
            
            generated_text = response['message']['content']
            
            # メモリに追加
            self.memory.append({
                "role": "assistant",
                "content": generated_text,
                "timestamp": datetime.now().isoformat()
            })
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Response generation error for {self.agent_id}: {e}")
            return self._get_fallback_response(context)
    
    def _get_fallback_response(self, context: Dict) -> str:
        """エラー時のフォールバック応答"""
        opponent_name = context.get('opponent_name', '相手')
        
        # キャラクターに応じた基本応答
        if 'high_school' in self.character_type:
            return f"{opponent_name}さん、それってすごく難しいですね...でも面白い！"
        elif 'office_worker' in self.character_type:
            return f"{opponent_name}さんの意見、実務的な観点から興味深いですね。"
        elif 'college_student' in self.character_type:
            return f"{opponent_name}さん、その視点は新鮮ですね。もう少し考えてみます。"
        else:
            return f"{opponent_name}さん、なるほど、そういう見方もありますね。"
    
    def get_character_info(self) -> Dict:
        """キャラクター情報を取得"""
        return {
            "id": self.agent_id,
            "name": self.character['name'],
            "type": self.character_type,
            "personality": self.character.get('personality', ''),
            "background": self.character.get('background', '')
        }
    
    def reset(self):
        """エージェントをリセット"""
        self.session_context = {}
        self.turn_directives = []
        self.memory = []
        logger.info(f"Agent {self.agent_id} reset")