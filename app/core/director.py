"""
Director（監督AI）クラス
対話の品質管理と戦略的介入を担当
"""

import json
import ollama
import re
from typing import Dict, List, Optional, Any, Tuple
import os

class Director:
    """対話を監督し、品質向上のための介入を行うAI"""
    
    def __init__(
        self,
        model_name: str = "qwen2.5:7b",
        temperature: float = 0.3
    ):
        """
        Args:
            model_name: 使用するOllamaモデル
            temperature: 生成時の温度（低めで安定した分析）
        """
        self.model_name = model_name
        self.temperature = temperature
        
        # 設定ファイルを読み込み
        self.strategies = self._load_strategies()
        self.prompt_templates = self._load_prompt_templates()
        
        # 分析履歴
        self.analysis_history = []
        self.intervention_count = 0
        
    def _load_strategies(self) -> Dict:
        """戦略設定を読み込み"""
        try:
            config_path = os.path.join("config", "strategies.json")
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: strategies.json not found at {config_path}")
            return self._get_default_strategies()
        except Exception as e:
            print(f"Error loading strategies: {e}")
            return self._get_default_strategies()
    
    def _get_default_strategies(self) -> Dict:
        """デフォルト戦略"""
        return {
            "discussion_strategies": {
                "deepening": {
                    "name": "議論の深化",
                    "interventions": ["具体例を挙げてください"]
                }
            },
            "discussion_phases": {
                "exploration": {
                    "name": "探索フェーズ",
                    "duration_turns": 5
                }
            }
        }
    
    def _load_prompt_templates(self) -> Dict:
        """プロンプトテンプレートを読み込み"""
        try:
            config_path = os.path.join("config", "prompt_templates.json")
            with open(config_path, "r", encoding="utf-8") as f:
                templates = json.load(f)
                return templates["director_prompt_template"]
        except FileNotFoundError:
            print(f"Warning: prompt_templates.json not found at {config_path}")
            return self._get_default_templates()
        except Exception as e:
            print(f"Error loading templates: {e}")
            return self._get_default_templates()
    
    def _get_default_templates(self) -> Dict:
        """デフォルトテンプレート"""
        return {
            "analysis": "対話を分析してください: {dialogue_history}",
            "strategy_selection": "戦略を選択してください: {analysis_result}",
            "quality_assessment": "品質を評価してください: {latest_response}"
        }
    
    def analyze_dialogue(
        self,
        dialogue_history: List[Dict],
        current_phase: str
    ) -> Dict[str, Any]:
        """対話の現状を分析"""
        
        # 対話履歴を文字列化
        history_text = self._format_dialogue_history(dialogue_history)
        
        # 分析プロンプトを構築
        prompt = self.prompt_templates["analysis"].format(
            dialogue_history=history_text,
            current_phase=current_phase
        )
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": self.temperature, "num_predict": 300}
            )
            
            content = response['message']['content']
            
            # JSON部分を抽出
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group(1))
            else:
                # フォールバック: 簡易解析
                analysis = self._fallback_analysis(dialogue_history)
            
            # 分析履歴に追加
            self.analysis_history.append(analysis)
            return analysis
            
        except Exception as e:
            print(f"Analysis error: {e}")
            return self._fallback_analysis(dialogue_history)
    
    def _fallback_analysis(self, dialogue_history: List[Dict]) -> Dict:
        """フォールバック分析（LLM失敗時）"""
        # 簡単なルールベース分析
        turn_count = len(dialogue_history)
        
        # 繰り返しチェック
        repetition = 0
        if turn_count > 2:
            recent_contents = [d['content'] for d in dialogue_history[-3:]]
            for i in range(len(recent_contents)-1):
                if recent_contents[i][:20] in recent_contents[i+1]:
                    repetition += 1
        
        return {
            "depth_level": min(3, turn_count // 3),
            "divergence": min(0.5, repetition * 0.2),
            "conflict_level": 0.3,
            "productivity": max(0.3, 1.0 - repetition * 0.3),
            "key_issues": ["分析未完了"],
            "strengths": ["対話継続中"],
            "dominant_pattern": "unknown"
        }
    
    def select_strategy(
        self,
        analysis_result: Dict,
        current_phase: str,
        turn_count: int
    ) -> Tuple[str, Dict]:
        """最適な戦略を選択し、指示を生成"""
        
        # 利用可能な戦略を取得
        available_strategies = self.strategies.get("discussion_strategies", {})
        phase_info = self.strategies.get("discussion_phases", {}).get(current_phase, {})
        phase_strategies = phase_info.get("preferred_strategies", list(available_strategies.keys()))
        
        # 戦略選択プロンプトを構築
        prompt = self.prompt_templates["strategy_selection"].format(
            analysis_result=json.dumps(analysis_result, ensure_ascii=False),
            current_phase=current_phase,
            available_strategies=json.dumps(
                {k: v["name"] for k, v in available_strategies.items()},
                ensure_ascii=False
            ),
            phase_strategies=", ".join(phase_strategies)
        )
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": self.temperature, "num_predict": 300}
            )
            
            content = response['message']['content']
            
            # JSON抽出
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                strategy_result = json.loads(json_match.group(1))
                selected_strategy = strategy_result.get("selected_strategy", "deepening")
                
                # 戦略が存在するか確認
                if selected_strategy not in available_strategies:
                    selected_strategy = phase_strategies[0] if phase_strategies else "deepening"
                
                instruction_data = {
                    "instruction": strategy_result.get(
                        "instruction_for_next_speaker",
                        self._get_random_intervention(selected_strategy)
                    ),
                    "attention_points": strategy_result.get("attention_points", []),
                    "expected_outcome": strategy_result.get("expected_outcome", "")
                }
                
            else:
                # フォールバック
                selected_strategy = self._select_strategy_by_rules(analysis_result, phase_strategies)
                instruction_data = {
                    "instruction": self._get_random_intervention(selected_strategy),
                    "attention_points": [],
                    "expected_outcome": ""
                }
                
        except Exception as e:
            print(f"Strategy selection error: {e}")
            # エラー時のフォールバック
            selected_strategy = phase_strategies[0] if phase_strategies else "deepening"
            instruction_data = {
                "instruction": self._get_random_intervention(selected_strategy),
                "attention_points": [],
                "expected_outcome": ""
            }
        
        self.intervention_count += 1
        return selected_strategy, instruction_data
    
    def _select_strategy_by_rules(
        self,
        analysis: Dict,
        preferred_strategies: List[str]
    ) -> str:
        """ルールベースで戦略を選択"""
        depth = analysis.get("depth_level", 2)
        divergence = analysis.get("divergence", 0.5)
        productivity = analysis.get("productivity", 0.5)
        
        # 生産性が低い場合
        if productivity < 0.3:
            return "perspective_shift"
        
        # 深さが浅い場合
        if depth < 2:
            return "deepening"
        
        # 発散しすぎている場合
        if divergence > 0.7:
            return "convergence"
        
        # デフォルトは推奨戦略から
        return preferred_strategies[0] if preferred_strategies else "deepening"
    
    def _get_random_intervention(self, strategy_name: str) -> str:
        """戦略に応じた介入指示を取得"""
        strategy = self.strategies.get("discussion_strategies", {}).get(strategy_name, {})
        interventions = strategy.get("interventions", ["建設的に議論を進めてください"])
        
        import random
        return random.choice(interventions)
    
    def evaluate_response(
        self,
        response: str,
        given_instruction: str,
        character_info: Dict
    ) -> Dict[str, Any]:
        """生成された応答の品質を評価"""
        
        prompt = self.prompt_templates["quality_assessment"].format(
            latest_response=response,
            given_instruction=given_instruction,
            character_info=json.dumps(character_info, ensure_ascii=False)
        )
        
        try:
            result = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": self.temperature, "num_predict": 200}
            )
            
            content = result['message']['content']
            
            # JSON抽出
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                evaluation = json.loads(json_match.group(1))
            else:
                evaluation = self._fallback_evaluation()
                
            return evaluation
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            return self._fallback_evaluation()
    
    def _fallback_evaluation(self) -> Dict:
        """フォールバック評価"""
        return {
            "instruction_compliance": 7,
            "character_consistency": 7,
            "contribution_quality": 6,
            "creativity": 6,
            "overall_score": 6.5,
            "improvement_suggestion": "継続して議論を深めてください"
        }
    
    def should_intervene_strongly(self, analysis: Dict) -> bool:
        """強い介入が必要か判定"""
        # 生産性が極端に低い
        if analysis.get("productivity", 0.5) < 0.2:
            return True
        
        # 深さが1で5ターン以上経過
        if analysis.get("depth_level", 2) == 1 and len(self.analysis_history) > 5:
            return True
        
        # 同じパターンが3回連続
        if len(self.analysis_history) >= 3:
            recent_patterns = [a.get("dominant_pattern") for a in self.analysis_history[-3:]]
            if len(set(recent_patterns)) == 1:
                return True
        
        return False
    
    def get_phase_transition_recommendation(
        self,
        current_phase: str,
        turn_count: int,
        analysis: Dict
    ) -> Optional[str]:
        """フェーズ移行の推奨を取得"""
        phases = self.strategies.get("discussion_phases", {})
        current_phase_info = phases.get(current_phase, {})
        expected_duration = current_phase_info.get("duration_turns", 5)
        
        # 期待ターン数を超過
        if turn_count >= expected_duration:
            # 次のフェーズを決定
            phase_order = ["exploration", "deepening", "convergence", "synthesis"]
            if current_phase in phase_order:
                current_index = phase_order.index(current_phase)
                if current_index < len(phase_order) - 1:
                    return phase_order[current_index + 1]
        
        # 早期移行が必要な場合
        if analysis.get("productivity", 0.5) < 0.3 and turn_count > 3:
            return "perspective_shift"  # 特別な介入
        
        return None
    
    def _format_dialogue_history(self, history: List[Dict]) -> str:
        """対話履歴を文字列化"""
        lines = []
        for item in history[-10:]:  # 最新10件のみ
            role = item.get("role", "不明")
            content = item.get("content", "")
            if len(content) > 100:
                content = content[:100] + "..."
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
    
    def get_statistics(self) -> Dict:
        """統計情報を取得"""
        if not self.analysis_history:
            return {}
        
        avg_depth = sum(a.get("depth_level", 0) for a in self.analysis_history) / len(self.analysis_history)
        avg_productivity = sum(a.get("productivity", 0) for a in self.analysis_history) / len(self.analysis_history)
        
        return {
            "total_interventions": self.intervention_count,
            "average_depth": avg_depth,
            "average_productivity": avg_productivity,
            "analysis_count": len(self.analysis_history)
        }
    
    def reset(self):
        """Directorをリセット"""
        self.analysis_history = []
        self.intervention_count = 0