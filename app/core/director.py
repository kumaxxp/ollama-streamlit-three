"""
Director AI - 自発的な介入判断を行う監督AI
LLMの能力を最大限活用し、プログラム的制御を最小限に
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class AutonomousDirector:
    """
    自発的な判断と介入を行うDirector AI
    プロンプトエンジニアリングによる高度な制御
    """
    
    def __init__(self, ollama_client, model_name: str = "qwen2.5:7b"):
        self.client = ollama_client
        self.model_name = model_name
        self.temperature = 0.3  # 一貫性重視
        
        # プロンプトテンプレートをJSONから読み込み
        self.load_prompts()
        
        # 介入履歴（学習用）
        self.intervention_history = []
        self.current_phase = "exploration"
        
        # 長さ分析用の履歴
        self.length_history = []
        self.target_length = "標準"  # デフォルトは標準
        
    def load_prompts(self):
        """Director用プロンプトをJSONから読み込み"""
        try:
            with open('config/director_prompts.json', 'r', encoding='utf-8') as f:
                self.prompts = json.load(f)
        except FileNotFoundError:
            logger.warning("director_prompts.json not found, using defaults")
            self.prompts = self._get_default_prompts()
    
    def _get_default_prompts(self) -> Dict:
        """デフォルトプロンプト（フォールバック用）"""
        return {
            "system_prompt": "あなたは対話の監督者です。必要に応じて介入してください。",
            "evaluation_prompt": "対話を評価し、介入の必要性を判断してください。"
        }
    
    def analyze_response_lengths(self, dialogue_context: List[Dict]) -> Dict:
        """
        対話の長さを分析
        
        Returns:
            {
                "average_length": float,
                "recent_trend": str,  # "increasing", "decreasing", "stable"
                "balance": str,  # "balanced", "imbalanced"
                "recommendation": str  # "簡潔", "標準", "詳細"
            }
        """
        if not dialogue_context:
            return {
                "average_length": 0,
                "recent_trend": "stable",
                "balance": "balanced",
                "recommendation": "標準"
            }
        
        # 最近5ターンの長さを分析
        recent_messages = dialogue_context[-5:] if len(dialogue_context) >= 5 else dialogue_context
        
        lengths = []
        agent_lengths = {"agent1": [], "agent2": []}
        
        for entry in recent_messages:
            if entry.get('speaker') != 'Director':
                message = entry.get('message', '')
                length = len(message)
                lengths.append(length)
                
                # エージェント別に記録
                speaker = entry.get('speaker', '')
                if 'やな' in speaker or 'さくら' in speaker or '田中' in speaker:
                    agent_lengths["agent1"].append(length)
                else:
                    agent_lengths["agent2"].append(length)
        
        if not lengths:
            return {
                "average_length": 0,
                "recent_trend": "stable",
                "balance": "balanced",
                "recommendation": "標準"
            }
        
        # 平均長さ
        avg_length = sum(lengths) / len(lengths)
        
        # トレンド分析
        if len(lengths) >= 3:
            recent_avg = sum(lengths[-2:]) / 2
            older_avg = sum(lengths[:-2]) / (len(lengths) - 2)
            if recent_avg > older_avg * 1.3:
                trend = "increasing"
            elif recent_avg < older_avg * 0.7:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        # バランス分析
        if agent_lengths["agent1"] and agent_lengths["agent2"]:
            avg1 = sum(agent_lengths["agent1"]) / len(agent_lengths["agent1"])
            avg2 = sum(agent_lengths["agent2"]) / len(agent_lengths["agent2"])
            ratio = max(avg1, avg2) / min(avg1, avg2) if min(avg1, avg2) > 0 else 1
            balance = "imbalanced" if ratio > 2 else "balanced"
        else:
            balance = "balanced"
        
        # 推奨を決定
        if avg_length > 250:
            recommendation = "簡潔"
        elif avg_length < 80:
            recommendation = "詳細"
        else:
            recommendation = "標準"
        
        # 履歴に追加
        self.length_history.append({
            "turn": len(dialogue_context),
            "average": avg_length,
            "trend": trend,
            "balance": balance
        })
        
        return {
            "average_length": avg_length,
            "recent_trend": trend,
            "balance": balance,
            "recommendation": recommendation
        }
    
    async def evaluate_dialogue(self, dialogue_context: List[Dict]) -> Dict:
        """
        対話を評価し、自発的に介入を判断（長さ制御を含む）
        
        Returns:
            {
                "intervention_needed": bool,
                "reason": str,
                "intervention_type": str,
                "message": str,
                "response_length_guide": str,
                "confidence": float
            }
        """
        # 長さ分析を実行
        length_analysis = self.analyze_response_lengths(dialogue_context)
        
        # 最近の対話を文脈として整形
        context_str = self._format_dialogue_context(dialogue_context)
        
        # 評価プロンプト構築（長さ情報を含む）
        evaluation_prompt = self._build_evaluation_prompt_with_length(
            context_str, 
            length_analysis
        )
        
        try:
            # LLMによる自発的判断
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.prompts["system_prompt"]},
                    {"role": "user", "content": evaluation_prompt}
                ],
                options={
                    "temperature": self.temperature,
                    "top_p": 0.9,
                    "seed": 42  # 再現性のため
                },
                format="json"
            )
            
            # レスポンスをパース
            result = json.loads(response['message']['content'])
            
            # 長さガイドが含まれていない場合はデフォルト値を設定
            if "response_length_guide" not in result:
                result["response_length_guide"] = length_analysis["recommendation"]
            
            # 現在の目標長さを更新
            self.target_length = result.get("response_length_guide", "標準")
            
            # 介入履歴に記録（学習用）
            if result.get("intervention_needed", False):
                self.intervention_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "type": result.get("intervention_type"),
                    "reason": result.get("reason"),
                    "phase": self.current_phase,
                    "length_guide": result.get("response_length_guide")
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Director evaluation error: {e}")
            return {
                "intervention_needed": False,
                "reason": "評価エラー",
                "intervention_type": "none",
                "message": "",
                "response_length_guide": "標準",
                "confidence": 0.0
            }
    
    def _build_evaluation_prompt_with_length(self, context: str, length_analysis: Dict) -> str:
        """長さ情報を含む評価プロンプトの構築"""
        return f"""
現在の対話フェーズ: {self.current_phase}

最近の対話内容:
{context}

対話の長さ分析:
- 平均文字数: {length_analysis['average_length']:.0f}
- 最近の傾向: {length_analysis['recent_trend']}
- バランス: {length_analysis['balance']}
- 現在の推奨: {length_analysis['recommendation']}

この対話を分析し、介入の必要性を自発的に判断してください。
特に応答が冗長または短すぎる場合は、長さ調整の介入を検討してください。

回答は必ず以下のJSON形式で:
{{
    "intervention_needed": true/false,
    "reason": "判断理由",
    "intervention_type": "質問投げかけ|要約|方向転換|深掘り|激励|仲裁|長さ調整|なし",
    "message": "介入する場合のメッセージ",
    "response_length_guide": "簡潔|標準|詳細|現状維持",
    "confidence": 0.0-1.0
}}
"""
    
    def _format_dialogue_context(self, dialogue_context: List[Dict]) -> str:
        """対話履歴を文字列に整形"""
        if not dialogue_context:
            return "（対話開始）"
        
        # 最新5ターンまでを取得
        recent = dialogue_context[-5:]
        formatted = []
        
        for entry in recent:
            speaker = entry.get('speaker', '不明')
            listener = entry.get('listener', '相手')
            message = entry.get('message', '')
            
            # Director介入は別形式で表示
            if speaker == 'Director':
                formatted.append(f"[監督介入] {message}")
            else:
                formatted.append(f"{speaker} → {listener}: 「{message}」")
        
        return "\n".join(formatted)
    
    def _build_evaluation_prompt(self, context: str) -> str:
        """評価用プロンプトの構築"""
        return f"""
現在の対話フェーズ: {self.current_phase}

最近の対話内容:
{context}

この対話を分析し、介入の必要性を自発的に判断してください。
あなたの判断基準に従って、最適な行動を選択してください。

回答は必ず以下のJSON形式で:
{{
    "intervention_needed": true/false,
    "reason": "判断理由",
    "intervention_type": "質問投げかけ|要約|方向転換|深掘り|激励|なし",
    "message": "介入する場合のメッセージ（相手を意識した自然な日本語で）",
    "confidence": 0.0-1.0の確信度
}}
"""
    
    def generate_opening_instruction(self, theme: str, agent_name: str) -> Dict:
        """
        対話開始時の指示生成
        """
        prompt = f"""
テーマ「{theme}」について、{agent_name}が最初の発言をします。
このエージェントへの効果的な開始指示を生成してください。

以下のJSON形式で回答:
{{
    "instruction": "具体的な指示",
    "tone": "推奨トーン（casual/formal/passionate等）",
    "focus_points": ["注目すべきポイント1", "ポイント2"]
}}
"""
        
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.prompts.get("instruction_system", "")},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0.5},
                format="json"
            )
            
            return json.loads(response['message']['content'])
            
        except Exception as e:
            logger.error(f"Opening instruction generation error: {e}")
            return {
                "instruction": f"{theme}について、あなたの視点から話を始めてください",
                "tone": "natural",
                "focus_points": ["自分の経験", "率直な意見"]
            }
    
    def update_phase(self, turn_count: int):
        """対話フェーズの更新"""
        if turn_count < 5:
            self.current_phase = "exploration"
        elif turn_count < 10:
            self.current_phase = "deepening"
        elif turn_count < 15:
            self.current_phase = "convergence"
        else:
            self.current_phase = "synthesis"
    
    def generate_length_instruction(self, target_length: str = "標準") -> str:
        """
        応答長の指示を生成
        
        Args:
            target_length: "簡潔", "標準", "詳細"
            
        Returns:
            エージェントへの長さ指示文字列
        """
        if target_length not in self.prompts.get("response_length_guidelines", {}):
            target_length = "標準"
        
        guideline = self.prompts["response_length_guidelines"][target_length]
        
        return f"""
【応答の長さについて】
{guideline['instruction']}
目安: {guideline['sentences']}
"""
    
    def get_intervention_stats(self) -> Dict:
        """介入統計を取得（分析用）"""
        if not self.intervention_history:
            return {"total": 0, "by_type": {}, "by_phase": {}}
        
        stats = {
            "total": len(self.intervention_history),
            "by_type": {},
            "by_phase": {}
        }
        
        for intervention in self.intervention_history:
            # タイプ別集計
            itype = intervention.get("type", "unknown")
            stats["by_type"][itype] = stats["by_type"].get(itype, 0) + 1
            
            # フェーズ別集計
            phase = intervention.get("phase", "unknown")
            stats["by_phase"][phase] = stats["by_phase"].get(phase, 0) + 1
        
        return stats
    
    def should_end_dialogue(self, dialogue_context: List[Dict], turn_count: int) -> Tuple[bool, str]:
        """
        対話終了判断（LLMによる自発的判断）
        """
        if turn_count < 10:  # 最低ターン数
            return False, ""
        
        context_str = self._format_dialogue_context(dialogue_context[-10:])
        
        prompt = f"""
以下の対話が十分な結論に達したか、または継続が困難か判断してください。

対話内容（最新10ターン）:
{context_str}

現在のターン数: {turn_count}

以下のJSON形式で回答:
{{
    "should_end": true/false,
    "reason": "終了/継続の理由"
}}
"""
        
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0.3},
                format="json"
            )
            
            result = json.loads(response['message']['content'])
            return result.get("should_end", False), result.get("reason", "")
            
        except Exception as e:
            logger.error(f"End dialogue evaluation error: {e}")
            return False, ""