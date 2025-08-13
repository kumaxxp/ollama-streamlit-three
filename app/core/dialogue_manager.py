"""
DialogueManager クラス
対話全体のオーケストレーションを担当
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Generator
from .agent import Agent
from .director import Director

class DialogueManager:
    """対話セッション全体を管理"""
    
    def __init__(
        self,
        theme: str,
        agent1_config: Dict,
        agent2_config: Dict,
        director_config: Optional[Dict] = None,
        max_turns: int = 20
    ):
        """
        Args:
            theme: 議論のテーマ
            agent1_config: エージェント1の設定 {"character": "...", "model": "...", "temperature": ...}
            agent2_config: エージェント2の設定
            director_config: Directorの設定 {"model": "...", "temperature": ...}
            max_turns: 最大ターン数
        """
        self.theme = theme
        self.max_turns = max_turns
        self.current_turn = 0
        self.current_phase = "exploration"
        self.dialogue_history = []
        self.phase_history = []
        
        # エージェントを初期化
        self.agent1 = Agent(
            agent_id="agent1",
            character_type=agent1_config.get("character", "philosophical_socrates"),
            model_name=agent1_config.get("model", "qwen2.5:7b"),
            temperature=agent1_config.get("temperature", 0.7)
        )
        
        self.agent2 = Agent(
            agent_id="agent2",
            character_type=agent2_config.get("character", "scientific_darwin"),
            model_name=agent2_config.get("model", "qwen2.5:7b"),
            temperature=agent2_config.get("temperature", 0.7)
        )
        
        # Directorを初期化
        director_config = director_config or {}
        self.director = Director(
            model_name=director_config.get("model", "qwen2.5:7b"),
            temperature=director_config.get("temperature", 0.3)
        )
        
        # フェーズ管理
        self.phases = self._load_phases()
        
        # セッションコンテキストを設定
        self._initialize_session()
        
    def _load_phases(self) -> Dict:
        """フェーズ設定を読み込み"""
        try:
            config_path = os.path.join("config", "strategies.json")
            with open(config_path, "r", encoding="utf-8") as f:
                strategies = json.load(f)
                return strategies.get("discussion_phases", {})
        except Exception as e:
            print(f"Error loading phases: {e}")
            return {
                "exploration": {"name": "探索", "duration_turns": 5}
            }
    
    def _initialize_session(self):
        """セッションを初期化"""
        # 現在のフェーズ情報
        phase_info = self.phases.get(self.current_phase, {})
        goal = phase_info.get("goal", "建設的な議論")
        
        # エージェントにコンテキストを設定
        self.agent1.set_session_context(self.theme, goal, self.current_phase)
        self.agent2.set_session_context(self.theme, goal, self.current_phase)
        
        # フェーズ履歴に記録
        self.phase_history.append({
            "phase": self.current_phase,
            "start_turn": 0
        })
    
    def run_turn(self, first_speaker: str = "agent1") -> Dict[str, Any]:
        """1ターンを実行"""
        turn_result = {
            "turn": self.current_turn,
            "phase": self.current_phase,
            "exchanges": [],
            "director_analysis": None,
            "director_strategy": None,
            "quality_scores": []
        }
        
        # 対話履歴から最後の発言を取得
        last_message = ""
        if self.dialogue_history:
            last_message = self.dialogue_history[-1]["content"]
        else:
            # 初回は開始メッセージ
            last_message = f"「{self.theme}」について議論を始めましょう。"
        
        # Directorが現状を分析
        if len(self.dialogue_history) > 0:
            analysis = self.director.analyze_dialogue(
                self.dialogue_history,
                self.current_phase
            )
            turn_result["director_analysis"] = analysis
            
            # 戦略を選択し、指示を生成
            strategy, instruction_data = self.director.select_strategy(
                analysis,
                self.current_phase,
                self.current_turn
            )
            turn_result["director_strategy"] = {
                "strategy": strategy,
                "instruction": instruction_data
            }
        else:
            # 初回はデフォルト指示
            instruction_data = {
                "instruction": "テーマについて、あなたの専門性を活かして論点を提示してください",
                "attention_points": ["明確な立場を示す", "具体的に述べる"],
                "expected_outcome": "議論の出発点を作る"
            }
            turn_result["director_strategy"] = {
                "strategy": "exploration",
                "instruction": instruction_data
            }
        
        # 話者を決定
        if first_speaker == "agent1":
            current_agent = self.agent1
            opponent_agent = self.agent2
        else:
            current_agent = self.agent2
            opponent_agent = self.agent1
        
        # エージェント1の発言
        current_agent.add_directive(
            instruction_data["instruction"],
            instruction_data.get("attention_points", [])
        )
        
        response1 = current_agent.generate_response(last_message)
        
        # 対話履歴に追加
        self.dialogue_history.append({
            "role": current_agent.agent_id,
            "name": current_agent.get_character_info()["name"],
            "content": response1,
            "turn": self.current_turn,
            "phase": self.current_phase
        })
        
        # 相手エージェントのメモリにも追加
        opponent_agent.add_to_memory(current_agent.agent_id, response1)
        
        # 品質評価
        quality1 = self.director.evaluate_response(
            response1,
            instruction_data["instruction"],
            current_agent.get_character_info()
        )
        turn_result["quality_scores"].append(quality1)
        
        # エージェント2の応答
        # 新しい分析と指示
        analysis2 = self.director.analyze_dialogue(
            self.dialogue_history,
            self.current_phase
        )
        strategy2, instruction_data2 = self.director.select_strategy(
            analysis2,
            self.current_phase,
            self.current_turn
        )
        
        opponent_agent.add_directive(
            instruction_data2["instruction"],
            instruction_data2.get("attention_points", [])
        )
        
        response2 = opponent_agent.generate_response(response1)
        
        # 対話履歴に追加
        self.dialogue_history.append({
            "role": opponent_agent.agent_id,
            "name": opponent_agent.get_character_info()["name"],
            "content": response2,
            "turn": self.current_turn,
            "phase": self.current_phase
        })
        
        # 相手エージェントのメモリにも追加
        current_agent.add_to_memory(opponent_agent.agent_id, response2)
        
        # 品質評価
        quality2 = self.director.evaluate_response(
            response2,
            instruction_data2["instruction"],
            opponent_agent.get_character_info()
        )
        turn_result["quality_scores"].append(quality2)
        
        # ターン結果に追加
        turn_result["exchanges"] = [
            {
                "speaker": current_agent.get_character_info()["name"],
                "content": response1,
                "instruction": instruction_data["instruction"]
            },
            {
                "speaker": opponent_agent.get_character_info()["name"],
                "content": response2,
                "instruction": instruction_data2["instruction"]
            }
        ]
        
        # フェーズ移行チェック
        next_phase = self.director.get_phase_transition_recommendation(
            self.current_phase,
            self.current_turn,
            analysis2
        )
        if next_phase and next_phase != self.current_phase:
            self._transition_phase(next_phase)
            turn_result["phase_transition"] = next_phase
        
        self.current_turn += 1
        
        return turn_result
    
    def run_turn_streaming(
        self,
        first_speaker: str = "agent1"
    ) -> Generator[Dict[str, Any], None, None]:
        """1ターンをストリーミングで実行（UIでの使用）"""
        
        # 分析フェーズ
        yield {"type": "status", "message": "Director が対話を分析中..."}
        
        # 対話履歴から最後の発言を取得
        last_message = ""
        if self.dialogue_history:
            last_message = self.dialogue_history[-1]["content"]
        else:
            last_message = f"「{self.theme}」について議論を始めましょう。"
        
        # Director分析
        analysis = None
        if len(self.dialogue_history) > 0:
            analysis = self.director.analyze_dialogue(
                self.dialogue_history,
                self.current_phase
            )
            yield {"type": "analysis", "data": analysis}
        
        # 戦略選択
        yield {"type": "status", "message": "戦略を選択中..."}
        
        if analysis:
            strategy, instruction_data = self.director.select_strategy(
                analysis,
                self.current_phase,
                self.current_turn
            )
        else:
            instruction_data = {
                "instruction": "テーマについて、あなたの専門性を活かして論点を提示してください",
                "attention_points": ["明確な立場を示す"],
                "expected_outcome": "議論の出発点を作る"
            }
            strategy = "exploration"
        
        yield {"type": "strategy", "data": {"strategy": strategy, "instruction": instruction_data}}
        
        # エージェント選択
        if first_speaker == "agent1":
            current_agent = self.agent1
            opponent_agent = self.agent2
        else:
            current_agent = self.agent2
            opponent_agent = self.agent1
        
        # エージェント1の発言
        yield {
            "type": "status",
            "message": f"{current_agent.get_character_info()['name']} が発言を準備中..."
        }
        
        current_agent.add_directive(
            instruction_data["instruction"],
            instruction_data.get("attention_points", [])
        )
        
        # ストリーミング応答
        response1 = ""
        for chunk in current_agent.generate_response(last_message, stream=True):
            response1 += chunk
            yield {
                "type": "response_chunk",
                "speaker": current_agent.get_character_info()["name"],
                "content": chunk
            }
        
        # 完了通知
        yield {
            "type": "response_complete",
            "speaker": current_agent.get_character_info()["name"],
            "content": response1
        }
        
        # 履歴更新
        self.dialogue_history.append({
            "role": current_agent.agent_id,
            "name": current_agent.get_character_info()["name"],
            "content": response1,
            "turn": self.current_turn,
            "phase": self.current_phase
        })
        opponent_agent.add_to_memory(current_agent.agent_id, response1)
        
        # エージェント2の準備
        yield {
            "type": "status",
            "message": f"{opponent_agent.get_character_info()['name']} が応答を準備中..."
        }
        
        # 新しい指示を生成
        analysis2 = self.director.analyze_dialogue(
            self.dialogue_history,
            self.current_phase
        )
        strategy2, instruction_data2 = self.director.select_strategy(
            analysis2,
            self.current_phase,
            self.current_turn
        )
        
        opponent_agent.add_directive(
            instruction_data2["instruction"],
            instruction_data2.get("attention_points", [])
        )
        
        # ストリーミング応答
        response2 = ""
        for chunk in opponent_agent.generate_response(response1, stream=True):
            response2 += chunk
            yield {
                "type": "response_chunk",
                "speaker": opponent_agent.get_character_info()["name"],
                "content": chunk
            }
        
        # 完了通知
        yield {
            "type": "response_complete",
            "speaker": opponent_agent.get_character_info()["name"],
            "content": response2
        }
        
        # 履歴更新
        self.dialogue_history.append({
            "role": opponent_agent.agent_id,
            "name": opponent_agent.get_character_info()["name"],
            "content": response2,
            "turn": self.current_turn,
            "phase": self.current_phase
        })
        current_agent.add_to_memory(opponent_agent.agent_id, response2)
        
        # フェーズ移行チェック
        next_phase = self.director.get_phase_transition_recommendation(
            self.current_phase,
            self.current_turn,
            analysis2
        )
        
        if next_phase and next_phase != self.current_phase:
            self._transition_phase(next_phase)
            yield {"type": "phase_transition", "new_phase": next_phase}
        
        self.current_turn += 1
        
        # ターン完了
        yield {"type": "turn_complete", "turn": self.current_turn}
    
    def _transition_phase(self, new_phase: str):
        """フェーズを移行"""
        self.current_phase = new_phase
        phase_info = self.phases.get(new_phase, {})
        goal = phase_info.get("goal", "建設的な議論")
        
        # エージェントに通知
        self.agent1.set_session_context(self.theme, goal, new_phase)
        self.agent2.set_session_context(self.theme, goal, new_phase)
        
        # 履歴に記録
        self.phase_history.append({
            "phase": new_phase,
            "start_turn": self.current_turn
        })
    
    def run_dialogue(self) -> List[Dict]:
        """対話全体を実行"""
        results = []
        
        while self.current_turn < self.max_turns:
            # ターンごとに話者を交代
            first_speaker = "agent1" if self.current_turn % 2 == 0 else "agent2"
            turn_result = self.run_turn(first_speaker)
            results.append(turn_result)
            
            # 強制終了条件チェック
            if self._should_end_dialogue(turn_result):
                break
        
        return results
    
    def _should_end_dialogue(self, turn_result: Dict) -> bool:
        """対話を終了すべきか判定"""
        # 品質が極端に低い場合
        if turn_result.get("quality_scores"):
            avg_quality = sum(
                s.get("overall_score", 0) for s in turn_result["quality_scores"]
            ) / len(turn_result["quality_scores"])
            if avg_quality < 3:
                return True
        
        # 同じ内容の繰り返し（最後の4発言をチェック）
        if len(self.dialogue_history) >= 4:
            recent = self.dialogue_history[-4:]
            contents = [d["content"][:50] for d in recent]
            if len(set(contents)) == 1:
                return True
        
        return False
    
    def save_dialogue(self, filename: Optional[str] = None) -> str:
        """対話を保存"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dialogue_{timestamp}.json"
        
        save_dir = os.path.join("data", "dialogues")
        os.makedirs(save_dir, exist_ok=True)
        
        filepath = os.path.join(save_dir, filename)
        
        # 保存データを構築
        save_data = {
            "theme": self.theme,
            "agents": {
                "agent1": self.agent1.get_character_info(),
                "agent2": self.agent2.get_character_info()
            },
            "dialogue_history": self.dialogue_history,
            "phase_history": self.phase_history,
            "director_statistics": self.director.get_statistics(),
            "total_turns": self.current_turn,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        return filepath
    
    def get_summary(self) -> Dict:
        """対話のサマリーを取得"""
        if not self.dialogue_history:
            return {}
        
        # Director統計
        director_stats = self.director.get_statistics()
        
        # フェーズごとのターン数
        phase_turns = {}
        for i, phase_change in enumerate(self.phase_history):
            phase = phase_change["phase"]
            start = phase_change["start_turn"]
            end = self.phase_history[i+1]["start_turn"] if i+1 < len(self.phase_history) else self.current_turn
            phase_turns[phase] = end - start
        
        return {
            "theme": self.theme,
            "total_turns": self.current_turn,
            "final_phase": self.current_phase,
            "phase_turns": phase_turns,
            "agent1": self.agent1.get_character_info(),
            "agent2": self.agent2.get_character_info(),
            "director_statistics": director_stats,
            "dialogue_length": len(self.dialogue_history)
        }
    
    def reset(self):
        """セッションをリセット"""
        self.current_turn = 0
        self.current_phase = "exploration"
        self.dialogue_history = []
        self.phase_history = []
        
        self.agent1.reset()
        self.agent2.reset()
        self.director.reset()
        
        self._initialize_session()