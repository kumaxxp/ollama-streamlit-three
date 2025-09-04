"""
Simple Director — 最小限のディレクター
目的: エージェント同士の会話が破綻しないよう、基本的なリズム/長さ/話法だけを指示する。
複雑な検証や外部検索は一切行わない。

提供インターフェイス（互換）:
- update_phase(turn_count)
- generate_opening_instruction(theme, first_speaker_name)
- async evaluate_dialogue(dialogue_context)
- generate_length_instruction(response_length_guide)
- should_end_dialogue(dialogue_history, turn_count)
- build_history_metrics_block(last_messages)
- get_intervention_stats()

AutonomousDirector エイリアスを維持。
"""

from __future__ import annotations
import json
import logging
import random
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SimpleDirector:
    def __init__(self, ollama_client=None, model_name: str = "qwen2.5:7b"):
        self.client = ollama_client
        self.model_name = model_name
        self.phase = "flow"
        self.turn_counter = 0
        self.participants: List[str] = []  # 表示名の出現順に2名
        self.stats = {
            "avg_len": 0.0,
            "q_ratio": 0.0,
            "last_speaker": None,
        }
        self._intervention_stats = {"rhythm_guides": 0}
        self.session_theme: Optional[str] = None

    # --- 公開API ---
    def update_phase(self, turn_count: int) -> None:
        self.phase = "wrap" if turn_count >= 16 else "flow"

    def generate_opening_instruction(self, theme: str, first_speaker_name: str) -> Dict[str, Any]:
        instruction = (
            f"テーマ『{theme}』について、挨拶は省き、自然な導入で語り始めてください。"
            f" {first_speaker_name}の相手が返しやすい具体例を1つ添え、2〜4文・200〜300文字で。"
            " 箇条書きや見出しは使わず、会話文で。"
        )
        focus = [
            "2〜4文・200〜300字",
            "相手が返しやすい1点だけ具体例",
            "挨拶や自己紹介はしない",
            "リスト・絵文字禁止",
        ]
        return {"instruction": instruction, "focus_points": focus}

    async def evaluate_dialogue(self, dialogue_context: List[Dict[str, Any]], mode: Optional[str] = None) -> Dict[str, Any]:
        # 入力の正規化（DialogueController/DialogueManager 両対応）
        msgs: List[Dict[str, str]] = []
        for m in dialogue_context or []:
            if not isinstance(m, dict):
                continue
            speaker = m.get("speaker") or m.get("role") or ""
            if speaker == "Director":
                continue
            # DialogueManager は 'message'、DialogueController は content を展開済
            message = m.get("message")
            if message is None:
                content = m.get("content")
                if isinstance(content, dict):
                    message = content.get("message", "")
                else:
                    message = str(content) if content is not None else ""
            listener = m.get("listener") or ""
            msgs.append({"speaker": str(speaker), "listener": str(listener), "message": str(message)})

        self.turn_counter += 1
        self._update_participants(msgs)
        avg_len, q_ratio = self._basic_metrics(msgs)
        self.stats.update({"avg_len": avg_len, "q_ratio": q_ratio, "last_speaker": (msgs[-1]["speaker"] if msgs else None)})

        plan = self._make_simple_plan(avg_len, q_ratio)
        self._intervention_stats["rhythm_guides"] += 1
        return {
            "intervention_needed": True,
            "reason": "basic_rhythm",
            "intervention_type": "length_tempo_speech_act",
            "message": json.dumps(plan, ensure_ascii=False),
            "response_length_guide": "簡潔" if avg_len > 120 else "標準",
            "confidence": 0.7,
            "director_debug": {"avg_len": avg_len, "q_ratio": q_ratio},
        }

    def generate_length_instruction(self, guide: str) -> str:
        guide = (guide or "").strip()
        mapping = {
            "簡潔": "応答の長さ: 100文字以内・最大2文",
            "標準": "応答の長さ: 140文字以内・最大3文",
            "詳細": "応答の長さ: 220文字以内・最大4文",
            "現状維持": "応答の長さは任意",
        }
        return mapping.get(guide, mapping["標準"])

    def should_end_dialogue(self, dialogue_history: List[Dict[str, Any]], turn_count: int) -> Tuple[bool, str]:
        # シンプル: wrap フェーズに入り、かつ直近で2回以上疑問がないなら終了示唆
        if turn_count >= 18:
            last_msgs = [h.get("message", "") if isinstance(h, dict) else "" for h in dialogue_history[-4:]]
            if last_msgs and not any(("?" in t or "？" in t) for t in last_msgs):
                return True, "wrap_phase_no_questions"
        return False, "continue"

    def build_history_metrics_block(self, last_messages: List[str]) -> str:
        # 古典っぽい引用の出現を数えるだけの簡易メトリクス
        classics = sum(len(re.findall(r"[「『][^」』]{4,30}[」』]", x or "")) for x in (last_messages or []))
        return (
            "[history-metrics]\n"
            f"classic_refs = {int(classics)}\n"
            "[/history-metrics]"
        )

    def get_intervention_stats(self) -> Dict[str, Any]:
        return dict(self._intervention_stats)

    # --- 内部ユーティリティ ---
    def _update_participants(self, msgs: List[Dict[str, str]]) -> None:
        if not msgs:
            return
        for m in msgs:
            s = m.get("speaker")
            if s and s != "Director" and s not in self.participants:
                self.participants.append(s)
            if len(self.participants) >= 2:
                break

    def _label_of(self, name: Optional[str]) -> str:
        if not name:
            return "A"
        if self.participants:
            if name == self.participants[0]:
                return "A"
            if len(self.participants) > 1 and name == self.participants[1]:
                return "B"
        return "A"

    def _basic_metrics(self, msgs: List[Dict[str, str]]) -> Tuple[float, float]:
        recent = [m for m in msgs][-6:]
        lengths = [len(m.get("message", "")) for m in recent]
        avg_len = (sum(lengths) / len(lengths)) if lengths else 0.0
        q_cnt = sum(1 for m in recent if ("?" in m.get("message", "") or "？" in m.get("message", "")))
        q_ratio = (q_cnt / len(recent)) if recent else 0.0
        return avg_len, q_ratio

    def _make_simple_plan(self, avg_len: float, q_ratio: float) -> Dict[str, Any]:
        # 長さガイド
        if avg_len > 140:
            max_chars, max_sent = 90, 2
        elif avg_len < 60:
            max_chars, max_sent = 140, 3
        else:
            max_chars, max_sent = 120, 3

        # 話法: 疑問が少なければ ask、多ければ answer
        speech_act = "ask" if q_ratio < 0.25 else "answer"

        # 次話者の指定は省略（Controller 側で交代制御）
        plan = {
            "turn_style": {
                "length": {"max_chars": max_chars, "max_sentences": max_sent},
                "preface": {"aizuchi": False, "aizuchi_list": [], "prob": 0.0},
                "speech_act": speech_act,
                "follow_up": "none",
                "ban": ["list_format", "long_intro", "speaker_label_prefix", "praise"],
            },
            "cadence": {"avoid_consecutive_monologues": True},
        }
        return plan


# 互換エイリアス
AutonomousDirector = SimpleDirector
