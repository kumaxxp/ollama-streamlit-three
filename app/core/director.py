"""
Director AI (v2) — Natural Conversation Director
目的: 議論の正しさではなく、人間的な会話のリズム(長さ/相槌/切り返し)を最優先で制御する。
- 介入は内容指図ではなく、テンポと尺と話法の指定(JSON)に限定する。
- 既存AutonomousDirectorを下位互換で拡張。

使用想定:
  director = NaturalConversationDirector(ollama_client, model_name="qwen2.5:7b")
  cmd = director.plan_next_turn(dialogue_context)
  # cmd は turn_style/cadence/closing_hint を含む JSON(dict)
  # これをそのまま Agent(A/B) に渡し、max_chars, max_sentences, aizuchi 等を実行させる。
"""

from __future__ import annotations
import json
import logging
import random
import re
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# 短い相槌と禁止語/冗長句の辞書(運用で増やす)
AIZUCHI_POOL = [
    "うん", "なるほど", "たしかに", "へぇ", "ふむ", "そうか", "ええ", "あー", "うーん"
]
PRAISE_WORDS = [
    "素晴らしい", "すごい", "勉強になります", "最高", "素敵", "称賛", "感謝します", "素晴らしかった"
]
LONG_INTRO_PAT = re.compile(r"^(ところで|まず|ちなみに|さて|えっと|あのー|まずは)[、,]\s*", re.UNICODE)
LIST_MARKERS_PAT = re.compile(r"^(?:[-*・]|\d+\.)\s", re.MULTILINE)

# KPI (会話の手触りを数値制約)
TURN_CHAR_MIN = 28
TURN_CHAR_MAX = 110
TURN_CHAR_MEDIAN_TARGET = 60
MAX_SENTENCES = 2
QUESTION_RATIO_TARGET = (0.35, 0.55)
AIZUCHI_PROB_DEFAULT = 0.4
MAX_CONSECUTIVE_BY_SPEAKER = 2

class NaturalConversationDirector:
    """会話のテンポ/尺/相槌を制御するディレクター。内容判断を最小化。"""

    def __init__(self, ollama_client=None, model_name: str = "qwen2.5:7b"):
        self.client = ollama_client  # v2ではLLM依存を極小化(無くても動く)
        self.model_name = model_name
        self.temperature = 0.2
        self.current_phase = "flow"  # flow→wrap
        self.turn_counter = 0
        # 検出した参加者名（会話ログの表示名）を A/B にマッピングするため保持
        self.participants: List[str] = []  # [nameA, nameB]
        # メトリクス
        self.metrics = {
            "avg_chars_last3": 0.0,
            "question_ratio": 0.0,
            "consecutive_by_last": 0,
            "last_speaker": None,            # 表示名
            "last_speaker_label": None,      # 'A' | 'B' | None
        }

    # ===== 公開API =====
    def plan_next_turn(self, dialogue_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """直近ログからテンポ指示JSONを生成する(LLM不要)。"""
        self.turn_counter += 1
        stats = self._analyze(dialogue_context)
        speaker = self._pick_speaker(stats)
        speech_act = self._pick_speech_act(stats)
        max_chars = self._decide_max_chars(stats)
        aizuchi = self._decide_aizuchi(stats)
        closing_hint = self._decide_closing(stats)
        # 禁止項目（会話汚染防止）
        ban_list = [
            "praise",
            "long_intro",
            "list_format",
            "other_character_name",
            "speaker_label_prefix",
        ]

        cmd = {
            "turn_style": {
                "speaker": speaker,
                "length": {"max_chars": max_chars, "max_sentences": MAX_SENTENCES},
                "preface": {
                    "aizuchi": aizuchi["on"],
                    "aizuchi_list": [aizuchi["word"]] if aizuchi["on"] else [],
                    "prob": aizuchi["prob"],
                },
                "speech_act": speech_act,  # ask|answer|reflect|agree_short|disagree_short|handoff
                "follow_up": "ask_feel" if speech_act == "ask" else "none",
                "ban": ban_list,
            },
            "cadence": {
                "avoid_consecutive_monologues": True,
                "enforce_question_ratio": list(QUESTION_RATIO_TARGET),
            },
            "closing_hint": closing_hint,
        }
        return cmd

    # ===== 下位互換API: フェーズ更新とオープニング指示 =====
    def update_phase(self, turn_count: int) -> None:
        """既存コード互換: ターン数に応じて内部フェーズを更新する。"""
        # シンプルな閾値で wrap へ移行（必要に応じて調整）
        self.current_phase = "wrap" if turn_count >= 16 else "flow"

    def generate_opening_instruction(self, theme: str, first_speaker_name: str) -> Dict[str, Any]:
        """既存コード互換: 最初の話者向けの開始指示を返す。"""
        instruction = (
            f"テーマ『{theme}』について、挨拶は省き、自然な導入で短く語り始めてください。"
            f" 相手（{first_speaker_name}の相手）が反応しやすい、具体的で一文～二文の投げかけを含めてください。"
            " 箇条書きは使わず、会話文で。"
        )
        focus_points = [
            "簡潔 (50-90文字)",
            "相手が返しやすい問いかけを添える",
            "挨拶や自己紹介はしない",
            "リスト/見出し/絵文字を使わない",
        ]
        return {"instruction": instruction, "focus_points": focus_points}

    # ===== 内部: 分析 =====
    def _analyze(self, dialogue_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        last3 = [m for m in dialogue_context if m.get("speaker") != "Director"][-3:]
        chars = [len(m.get("message", "")) for m in last3]
        avg_chars_last3 = sum(chars)/len(chars) if chars else 0

        # 質問検出: 末尾? or 疑問語を含む
        def is_question(text: str) -> bool:
            return text.strip().endswith("？") or any(q in text for q in ["？", "?", "どう思う", "なぜ", "理由", "どこ", "いつ", "だれ", "何"])
        last10 = [m for m in dialogue_context if m.get("speaker") != "Director"][-10:]
        q_cnt = sum(1 for m in last10 if is_question(m.get("message", "")))
        question_ratio = (q_cnt/len(last10)) if last10 else 0.0

        # 連続独演
        # 参加者名を検出して固定（先に登場した2名を参加者とする）
        if dialogue_context:
            names: List[str] = []
            for m in dialogue_context:
                s = m.get("speaker")
                if s and s != "Director" and s not in names:
                    names.append(s)
                if len(names) >= 2:
                    break
            if names:
                # 既存 participants を温存しつつ更新
                if not self.participants:
                    self.participants = names[:2]
                else:
                    # 不足があれば補完
                    for n in names:
                        if n not in self.participants and len(self.participants) < 2:
                            self.participants.append(n)

        last_speaker = self.metrics["last_speaker"]
        current_last = last3[-1]["speaker"] if last3 else None
        if current_last == last_speaker and current_last is not None:
            consecutive = self.metrics["consecutive_by_last"] + 1
        else:
            consecutive = 1
        # A/B ラベルに正規化
        last_label = None
        if current_last:
            last_label = self._label_of(current_last)
        # 更新
        self.metrics.update({
            "avg_chars_last3": avg_chars_last3,
            "question_ratio": question_ratio,
            "consecutive_by_last": consecutive,
            "last_speaker": current_last,
            "last_speaker_label": last_label,
        })
        return dict(self.metrics)

    # ===== 内部: 方針決定 =====
    def _pick_speaker(self, stats: Dict[str, Any]) -> str:
        # 連続独演を禁止
        if stats["consecutive_by_last"] >= MAX_CONSECUTIVE_BY_SPEAKER and (stats.get("last_speaker_label") or stats.get("last_speaker")):
            return self._other(stats.get("last_speaker_label") or stats.get("last_speaker"))  # 強制交代
        # 基本は交互
        return self._other(stats.get("last_speaker_label") or stats.get("last_speaker")) if stats.get("last_speaker") else "A"

    def _pick_speech_act(self, stats: Dict[str, Any]) -> str:
        lo, hi = QUESTION_RATIO_TARGET
        if stats["question_ratio"] < lo:
            return "ask"
        # 長尺が続くなら短い相づち系
        if stats["avg_chars_last3"] > 90:
            return random.choice(["reflect", "agree_short", "handoff"])
        return random.choice(["answer", "reflect", "agree_short", "disagree_short"])

    def _decide_max_chars(self, stats: Dict[str, Any]) -> int:
        if stats["avg_chars_last3"] > 90:
            return 70
        if stats["avg_chars_last3"] < 40:
            return 90  # 少し伸ばす
        return 80  # デフォルト

    def _decide_aizuchi(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        on = random.random() < AIZUCHI_PROB_DEFAULT
        return {"on": on, "word": random.choice(AIZUCHI_POOL), "prob": AIZUCHI_PROB_DEFAULT}

    def _decide_closing(self, stats: Dict[str, Any]) -> str:
        # 単純規則: ターンが進み、質問比率が閾値内で、新情報が減ってきたら締めに寄せる
        if self.turn_counter >= 14 and QUESTION_RATIO_TARGET[0] <= stats["question_ratio"] <= QUESTION_RATIO_TARGET[1]:
            return "小まとめして交代"
        if self.turn_counter >= 18:
            return "締めに向かう"
        return "続ける"

    def _other(self, s: Optional[str]) -> str:
        """与えられた話者（表示名または 'A'/'B'）の反対側ラベルを返す。"""
        if s is None:
            return "A"
        label = s
        if s not in ("A", "B"):
            label = self._label_of(s)
        return "B" if label == "A" else "A"

    def _label_of(self, name: str) -> str:
        """表示名から 'A' または 'B' を返す（未知は 'A'）。"""
        if not name:
            return "A"
        if self.participants:
            if len(self.participants) >= 1 and name == self.participants[0]:
                return "A"
            if len(self.participants) >= 2 and name == self.participants[1]:
                return "B"
        # 不明な名前の場合は A を既定に
        return "A"

    # ===== 参考: 既存 evaluate_dialogue の互換API(必要なら使用) =====
    async def evaluate_dialogue(self, dialogue_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """下位互換: 旧API呼び出し箇所のための非同期関数。v2では turn_style を返す。"""
        cmd = self.plan_next_turn(dialogue_context)
        return {
            "intervention_needed": True,
            "reason": "rhythm_control",
            "intervention_type": "length_tempo_speech_act",
            "message": json.dumps(cmd, ensure_ascii=False),  # 互換のため文字列化
            "response_length_guide": "簡潔",
            "confidence": 0.8,
        }

    # ===== Judgeユーティリティ(任意) =====
    @staticmethod
    def judge_text(text: str, max_chars: int = TURN_CHAR_MAX, max_sentences: int = MAX_SENTENCES) -> Dict[str, Any]:
        violations = []
        # 長さ
        if len(text) > max_chars:
            violations.append("too_long")
        # 文数
        sentences = re.split(r"[。.!?？！]+", text.strip())
        sentences = [s for s in sentences if s]
        if len(sentences) > max_sentences:
            violations.append("too_many_sentences")
        # リスト
        if LIST_MARKERS_PAT.search(text):
            violations.append("list_detected")
        # 称賛
        if any(w in text for w in PRAISE_WORDS):
            violations.append("praise_used")
        # 冗長書き出し
        if LONG_INTRO_PAT.search(text):
            violations.append("long_intro")
        return {"ok": not violations, "violations": violations}

    @staticmethod
    def auto_repair(text: str, max_chars: int = 90) -> str:
        # 冗長導入句を削除
        text = LONG_INTRO_PAT.sub("", text)
        # 箇条書き記号を削除
        text = re.sub(r"^(?:[-*・]|\d+\.)\s*", "", text, flags=re.MULTILINE)
        # 称賛語を中立語に置換
        for w in PRAISE_WORDS:
            text = text.replace(w, "(省略)")
        # 末尾刈り
        if len(text) > max_chars:
            text = text[:max_chars].rstrip("、, ")
        return text

# 既存クラス名に合わせたエイリアス(置換コスト最小化)
AutonomousDirector = NaturalConversationDirector
