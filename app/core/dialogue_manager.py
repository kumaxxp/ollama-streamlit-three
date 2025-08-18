"""
Dialogue Manager - 対話フロー全体の管理
自己対話を防ぎ、必ず交互に発言するよう制御
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import asyncio

from .agent import Agent
from .director import AutonomousDirector

logger = logging.getLogger(__name__)

class DialogueManager:
    """
    対話全体の流れを管理
    エージェント間の適切な対話を保証
    """
    
    def __init__(self, ollama_client, director_model: str = "qwen2.5:7b"):
        self.client = ollama_client
        self.director = AutonomousDirector(ollama_client, director_model)
        
        # エージェント管理
        self.agent1: Optional[Agent] = None
        self.agent2: Optional[Agent] = None
        self.current_speaker: Optional[Agent] = None
        self.current_listener: Optional[Agent] = None
        
        # 対話状態
        self.dialogue_history: List[Dict] = []
        self.theme = ""
        self.turn_count = 0
        self.is_running = False
        
        # 設定
        self.max_turns = 20
        self.enable_director = True
        
    def initialize(self, theme: str, agent1_config: Dict, agent2_config: Dict):
        """
        対話セッションの初期化
        
        Args:
            theme: 議論テーマ
            agent1_config: エージェント1の設定
            agent2_config: エージェント2の設定
        """
        self.theme = theme
        self.dialogue_history = []
        self.turn_count = 0
        
        # デバッグ情報
        logger.info(f"Initializing dialogue with theme: {theme}")
        logger.info(f"Agent1 config: {agent1_config}")
        logger.info(f"Agent2 config: {agent2_config}")
        
        # エージェント作成
        self.agent1 = Agent(
            agent_id="agent1",
            character_type=agent1_config.get('character_type'),
            model_name=agent1_config.get('model', 'qwen2.5:7b'),
            temperature=agent1_config.get('temperature', 0.7),
            ollama_client=self.client
        )
        
        self.agent2 = Agent(
            agent_id="agent2",
            character_type=agent2_config.get('character_type'),
            model_name=agent2_config.get('model', 'qwen2.5:7b'),
            temperature=agent2_config.get('temperature', 0.7),
            ollama_client=self.client
        )
        
        # キャラクター名を確認
        logger.info(f"Agent1 character name: {self.agent1.character.get('name', 'Unknown')}")
        logger.info(f"Agent2 character name: {self.agent2.character.get('name', 'Unknown')}")
        
        # セッションコンテキスト設定
        self.agent1.set_session_context(theme, "建設的な議論", "exploration")
        self.agent2.set_session_context(theme, "建設的な議論", "exploration")
        
        # 最初の話者をランダムに決定（または agent1 固定）
        self.current_speaker = self.agent1
        self.current_listener = self.agent2
        
        logger.info(f"Dialogue initialized - Theme: {theme}")
    
    async def run_turn(self) -> Dict:
        """
        1ターンの対話を実行
        必ず話者と聞き手を交代させる
        
        Returns:
            {
                "speaker": str,
                "listener": str,
                "message": str,
                "director_intervention": Optional[Dict]
            }
        """
        self.turn_count += 1
        self.director.update_phase(self.turn_count)
        
        # 話者と聞き手を明確に設定
        speaker = self.current_speaker
        listener = self.current_listener
        
        logger.info(f"Turn {self.turn_count}: {speaker.character['name']} → {listener.character['name']}")
        
        # 最初のターンの場合、開始指示を生成
        if self.turn_count == 1:
            opening_instruction = self.director.generate_opening_instruction(
                self.theme, 
                speaker.character['name']
            )
            speaker.add_directive(
                opening_instruction['instruction'],
                opening_instruction['focus_points']
            )
            
            # 最初の発言を生成
            message = await self._generate_first_message(speaker)
        else:
            # 前の発言への応答を生成
            last_message = self.dialogue_history[-1]['message'] if self.dialogue_history else ""
            message = await self._generate_response(speaker, listener, last_message)
        
        # 対話履歴に追加
        turn_data = {
            "turn": self.turn_count,
            "speaker": speaker.character['name'],
            "listener": listener.character['name'],
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        self.dialogue_history.append(turn_data)
        
        # Director による評価と介入判断
        director_result = None
        if self.enable_director and self.turn_count > 1:
            director_result = await self.director.evaluate_dialogue(self.dialogue_history)
            
            if director_result.get("intervention_needed", False):
                # 介入メッセージを履歴に追加
                intervention_data = {
                    "turn": self.turn_count,
                    "speaker": "Director",
                    "listener": "両者",
                    "message": director_result['message'],
                    "intervention_type": director_result['intervention_type'],
                    "timestamp": datetime.now().isoformat()
                }
                self.dialogue_history.append(intervention_data)
                
                # 次の発言者に介入内容を伝える
                self._apply_director_intervention(listener, director_result)
        
        # 話者と聞き手を交代
        self.current_speaker = listener
        self.current_listener = speaker
        
        # 結果を返す
        result = turn_data.copy()
        if director_result and director_result.get("intervention_needed"):
            result["director_intervention"] = director_result
        
        return result
    
    async def _generate_first_message(self, speaker: Agent) -> str:
        """最初の発言を生成"""
        prompt = f"""
あなたは{speaker.character['name']}です。

テーマ「{self.theme}」について、最初の発言をしてください。

あなたの性格と背景:
{json.dumps(speaker.character, ensure_ascii=False, indent=2)}

注意事項:
- 挨拶（こんにちは、よろしく等）は不要。すぐにテーマについて話し始める
- 自然な導入から始めてください
- あなたらしい視点で話してください
- 相手がいることを意識した発言にしてください
- 箇条書きや見出しは使わず、普通の会話文で話す
- マークダウン記号は使わない
 - 2〜4文で、200-300文字を目安に（やや詳しく）
"""
        
        try:
            response = self.client.chat(
                model=speaker.model_name,
                messages=[
                    {"role": "system", "content": speaker._build_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                options={
                    "temperature": speaker.temperature,
                    "top_p": 0.95,
                    # 安全な既定: KVキャッシュ/VRAM圧迫を抑える
                    "num_ctx": 4096,
                    "num_batch": 128,
                },
                stream=False
            )
            
            return response['message']['content']
            
        except Exception as e:
            logger.error(f"First message generation error: {e}")
            return f"こんにちは。{self.theme}について話しましょう。"
    
    async def _generate_response(self, speaker: Agent, listener: Agent, last_message: str) -> str:
        """
        相手の発言への応答を生成
        必ず相手に向けた発言にする
        """
        # 最近の文脈を取得（最新3ターン）
        recent_context = self._get_recent_context(3)
        
        prompt = f"""
あなたは{speaker.character['name']}です。
{listener.character['name']}さんとの対話中です。

これまでの文脈:
{recent_context}

{listener.character['name']}さんの最新の発言:
「{last_message}」

この発言に対して、{listener.character['name']}さんに向けて応答してください。

重要な指示:
- 必ず{listener.character['name']}さんに向けて話してください
- 自分（{speaker.character['name']}）に話しかけないでください
- 対話として自然な応答を心がけてください
- あなたの性格や立場を保ちながら応答してください
- 挨拶は不要です。相手の発言に直接応答してください
- 箇条書きや見出しは使わず、自然な会話文で話してください

あなたの性格設定:
{json.dumps(speaker.character, ensure_ascii=False, indent=2)}

【応答形式】
マークダウン記号（##、**、-、・など）は使わず、普通の会話として応答してください。
"""
        
        try:
            response = self.client.chat(
                model=speaker.model_name,
                messages=[
                    {"role": "system", "content": speaker._build_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                options={
                    "temperature": speaker.temperature,
                    "top_p": 0.95,
                    "seed": None,  # 多様性のため
                    # 安全な既定: KVキャッシュ/VRAM圧迫を抑える
                    "num_ctx": 4096,
                    "num_batch": 128,
                },
                stream=False
            )
            
            generated_message = response['message']['content']
            
            # 自己言及チェック（念のため）
            if self._is_self_reference(generated_message, speaker.character['name']):
                logger.warning(f"Self-reference detected, regenerating...")
                # 再生成を試みる
                return await self._generate_response(speaker, listener, last_message)
            
            return generated_message
            
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return f"{listener.character['name']}さん、それは興味深い視点ですね。"
    
    def _get_recent_context(self, num_turns: int = 3) -> str:
        """最近の対話文脈を取得"""
        if not self.dialogue_history:
            return "（これが最初の発言です）"
        
        recent = self.dialogue_history[-num_turns:]
        context_lines = []
        
        for entry in recent:
            speaker = entry.get('speaker')
            listener = entry.get('listener', '')
            message = entry.get('message', '')
            
            if speaker == 'Director':
                context_lines.append(f"[監督からの介入] {message}")
            else:
                if listener:
                    context_lines.append(f"{speaker} → {listener}: 「{message}」")
                else:
                    context_lines.append(f"{speaker}: 「{message}」")
        
        return "\n".join(context_lines)
    
    def _is_self_reference(self, message: str, speaker_name: str) -> bool:
        """
        自己言及をチェック
        （例：「私、太郎は太郎に言います」のような不自然な発言を検出）
        """
        # 簡易的なチェック
        if f"{speaker_name}に" in message and f"私は{speaker_name}" in message:
            return True
        if f"{speaker_name}さんに" in message and f"私、{speaker_name}" in message:
            return True
        
        return False
    
    def _apply_director_intervention(self, next_speaker: Agent, intervention: Dict):
        """Director の介入を次の発言者に適用（長さ指示を含む）"""
        instruction = intervention.get('message', '')
        intervention_type = intervention.get('intervention_type', '')
        response_length_guide = intervention.get('response_length_guide', '標準')
        
        # 長さ指示を生成
        length_instruction = self.director.generate_length_instruction(response_length_guide)
        
        # 介入タイプに応じた指示を生成
        if intervention_type == "長さ調整":
            next_speaker.add_directive(
                f"{instruction}\n{length_instruction}",
                ["応答の長さを調整", f"{response_length_guide}で応答"]
            )
        elif intervention_type == "質問投げかけ":
            next_speaker.add_directive(
                f"Directorからの質問を考慮してください: {instruction}\n{length_instruction}",
                ["新しい視点を提供", "質問に答える", f"{response_length_guide}で応答"]
            )
        elif intervention_type == "要約":
            next_speaker.add_directive(
                f"これまでの議論を踏まえて、発展的な意見を述べてください\n{length_instruction}",
                ["要約を活用", "次のステップを提案", f"{response_length_guide}で応答"]
            )
        elif intervention_type == "方向転換":
            next_speaker.add_directive(
                f"{instruction}\n{length_instruction}",
                ["新しい角度から", "視点を変えて", f"{response_length_guide}で応答"]
            )
        elif intervention_type == "深掘り":
            next_speaker.add_directive(
                f"この点を深く掘り下げてください: {instruction}\n{length_instruction}",
                ["具体例を挙げる", "詳細に説明", f"{response_length_guide}で応答"]
            )
        else:
            # その他の介入でも長さ指示を追加
            if response_length_guide != "現状維持":
                next_speaker.add_directive(
                    length_instruction,
                    [f"{response_length_guide}で応答"]
                )
    
    async def run_dialogue(self, max_turns: Optional[int] = None) -> List[Dict]:
        """
        対話全体を実行
        
        Args:
            max_turns: 最大ターン数（Noneの場合はデフォルト値）
        
        Returns:
            対話履歴
        """
        if max_turns:
            self.max_turns = max_turns
        
        self.is_running = True
        
        try:
            while self.turn_count < self.max_turns and self.is_running:
                # 1ターン実行
                turn_result = await self.run_turn()
                
                # 終了判断
                if self.turn_count >= 10:  # 最低10ターン後
                    should_end, reason = self.director.should_end_dialogue(
                        self.dialogue_history, 
                        self.turn_count
                    )
                    if should_end:
                        logger.info(f"Dialogue ended by Director: {reason}")
                        break
                
                # 少し待機（API負荷軽減）
                await asyncio.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Dialogue execution error: {e}")
        finally:
            self.is_running = False
        
        return self.dialogue_history
    
    def stop_dialogue(self):
        """対話を停止"""
        self.is_running = False
        logger.info("Dialogue stop requested")
    
    def get_summary(self) -> Dict:
        """対話のサマリーを取得"""
        if not self.dialogue_history:
            return {"status": "no_dialogue"}
        
        return {
            "theme": self.theme,
            "total_turns": self.turn_count,
            "participants": [
                self.agent1.character['name'] if self.agent1 else "Unknown",
                self.agent2.character['name'] if self.agent2 else "Unknown"
            ],
            "director_interventions": len([
                h for h in self.dialogue_history 
                if h.get('speaker') == 'Director'
            ]),
            "start_time": self.dialogue_history[0].get('timestamp'),
            "end_time": self.dialogue_history[-1].get('timestamp')
        }
    
    def save_dialogue(self, filepath: str):
        """対話を保存"""
        save_data = {
            "summary": self.get_summary(),
            "dialogue": self.dialogue_history,
            "director_stats": self.director.get_intervention_stats()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Dialogue saved to {filepath}")