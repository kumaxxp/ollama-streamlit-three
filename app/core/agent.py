"""
Agent - 対話エージェントの実装
キャラクター設定に基づく自然な対話生成
"""

import json
import logging
import os
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
        import os
        
        # 複数のパスを試す
        possible_paths = [
            'config/characters.json',
            './config/characters.json',
            '../config/characters.json',
            '../../config/characters.json',
            os.path.join(os.path.dirname(__file__), '../../config/characters.json')
        ]
        
        characters_data = None
        used_path = None
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        characters_data = json.load(f)
                        used_path = path
                        logger.info(f"Characters loaded from: {path}")
                        break
                except Exception as e:
                    logger.error(f"Error loading from {path}: {e}")
                    continue
        
        if not characters_data:
            logger.error(f"characters.json not found in any of these paths: {possible_paths}")
            logger.error(f"Current working directory: {os.getcwd()}")
            return self._get_default_character()
        
        # キャラクターを取得
        if 'characters' in characters_data:
            if character_type in characters_data['characters']:
                logger.info(f"Character '{character_type}' loaded successfully")
                character = characters_data['characters'][character_type]

                # 外部ファイル参照があれば読み込んで prompt_template にセットする
                if 'prompt_template_file' in character:
                    # 可能な相対パス候補: characters.json のディレクトリ基準、およびワークディレクトリ基準
                    base_dirs = []
                    if used_path:
                        base_dirs.append(os.path.dirname(os.path.abspath(used_path)))
                    base_dirs.extend([os.getcwd(), os.path.abspath('.')])

                    file_path = character.get('prompt_template_file')
                    found = False
                    for base in base_dirs:
                        candidate = os.path.join(base, file_path)
                        if os.path.exists(candidate):
                            try:
                                with open(candidate, 'r', encoding='utf-8') as pf:
                                    character['prompt_template'] = pf.read()
                                logger.info(f"Loaded prompt template from file: {candidate}")
                                found = True
                                break
                            except Exception as e:
                                logger.error(f"Failed to read prompt template file {candidate}: {e}")
                                continue
                    if not found:
                        # そのままファイルパスを試す
                        try:
                            if os.path.exists(file_path):
                                with open(file_path, 'r', encoding='utf-8') as pf:
                                    character['prompt_template'] = pf.read()
                                logger.info(f"Loaded prompt template from file: {file_path}")
                                found = True
                        except Exception as e:
                            logger.error(f"Failed to read prompt template file {file_path}: {e}")

                return character
            else:
                logger.warning(f"Character type '{character_type}' not found in characters.json")
                logger.warning(f"Available characters: {list(characters_data['characters'].keys())}")
                return self._get_default_character()
        else:
            logger.error("'characters' key not found in JSON")
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
        """システムプロンプトを構築（UIで表示するためprintは行わない）"""
        # キャラクターのプロンプトテンプレートを使用
        if 'prompt_template' in self.character:
            template = self.character['prompt_template']
            system_prompt = template.format(
                name=self.character['name'],
                personality=self.character.get('personality', ''),
                speaking_style=self.character.get('speaking_style', ''),
                background=self.character.get('background', ''),
                values=self.character.get('values', '')
            )
            return system_prompt
        # テンプレートがない場合は標準形式
        system_prompt = (
            f"あなたは{self.character['name']}です。\n"
            f"\n"
            f"【基本設定】\n"
            f"性格: {self.character.get('personality', '未設定')}\n"
            f"話し方: {self.character.get('speaking_style', '自然な日本語')}\n"
            f"背景: {self.character.get('background', '一般的な背景')}\n"
            f"価値観: {self.character.get('values', '建設的な対話')}\n"
            f"\n"
            f"【重要な指示】\n"
            f"- あなたのキャラクターを一貫して保ってください\n"
            f"- 相手の発言をよく聞き、それに応答してください\n"
            f"- 自然な対話を心がけてください\n"
            f"- 議論を建設的に進めてください\n"
        )
        return system_prompt

    # （不要な日本語生文字列を削除済み）
    
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
        
        # 最新のDirective（長さ指示を含む）
        if self.turn_directives:
            latest_directive = self.turn_directives[-1]
            prompt_parts.append(f"\n【今回の注意点】")
            prompt_parts.append(latest_directive['instruction'])
            if latest_directive['attention_points']:
                prompt_parts.append("- " + "\n- ".join(latest_directive['attention_points']))
            
            # 長さ指示が含まれているかチェック
            length_guidance = None
            for point in latest_directive['attention_points']:
                if "簡潔" in point:
                    length_guidance = "50-100文字程度で簡潔に"
                elif "詳細" in point:
                    length_guidance = "200-300文字程度で詳しく"
                elif "標準" in point:
                    length_guidance = "100-200文字程度で"
            
            if length_guidance:
                prompt_parts.append(f"\n【応答の長さ】{length_guidance}")
        
        # 応答指示
        prompt_parts.append(f"\n上記を踏まえて、{opponent_name}に対して{self.character['name']}として応答してください。")
        prompt_parts.append("自然で、あなたらしい発言を心がけてください。")
        
        # フォーマット指示を追加
        prompt_parts.append("\n【応答形式の注意】")
        prompt_parts.append("- 挨拶（こんにちは等）は不要です。すぐに本題に入ってください")
        prompt_parts.append("- 箇条書きや見出しは使わず、自然な会話文で応答してください")
        prompt_parts.append("- マークダウン記号（##、**、-、・など）は使わないでください")
        prompt_parts.append("- 普通に会話するような、流れるような文章で話してください")
        
        return "\n".join(prompt_parts)
    
    async def generate_response(self, context: Dict, system_prompt: Optional[str] = None, user_prompt: Optional[str] = None) -> str:
        """
        応答を生成
        
        Args:
            context: 対話コンテキスト
        
        Returns:
            生成された応答
        """
        # allow caller to provide pre-built prompts (to avoid duplicate building and to expose them to UI)
        if user_prompt is None:
            prompt = self.build_prompt(context)
        else:
            prompt = user_prompt

        if system_prompt is None:
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
            # Return structured error info so controller/UI can surface it
            return {
                "error": True,
                "message": self._get_fallback_response(context),
                "detail": str(e)
            }
    
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