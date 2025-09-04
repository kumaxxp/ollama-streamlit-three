"""
Natural Conversation Director — minimal and pragmatic

目的: 会話のテンポと長さを整えることを最優先に、必要最小限の“軽量ファクトチェック”
    （原子主張の抽出→Webスニペット照会→不足指摘）だけを実装したシンプル版。

方針:
- インターフェース互換（AutonomousDirector / evaluate_dialogue など）は維持。
- 失敗に強く、常にリズム制御へフォールバックできる設計。
"""

from __future__ import annotations
import json
import logging
import random
import re
from typing import Dict, List, Optional, Tuple, Any
from .search_adapter import MCPWebSearchAdapter

logger = logging.getLogger(__name__)

# 短い相槌と禁止語/冗長句の辞書
AIZUCHI_POOL = [
    "うん", "なるほど", "たしかに", "へぇ", "ふむ", "そうか", "ええ", "あー", "うーん"
]
PRAISE_WORDS = [
    "素晴らしい", "すごい", "勉強になります", "最高", "素敵", "称賛", "感謝します", "素晴らしかった"
]
LONG_INTRO_PAT = re.compile(r"^(ところで|まず|ちなみに|さて|えっと|あのー|まずは)[、,]\s*", re.UNICODE)
LIST_MARKERS_PAT = re.compile(r"^(?:[-*・]|\d+\.)\s", re.MULTILINE)

# KPI (会話の手触りを数値制約)
TURN_CHAR_MAX = 160
TURN_CHAR_MEDIAN_TARGET = 85
MAX_SENTENCES = 3
QUESTION_RATIO_TARGET = (0.30, 0.55)
AIZUCHI_PROB_DEFAULT = 0.35
MAX_CONSECUTIVE_BY_SPEAKER = 2


class NaturalConversationDirector:
    """会話のテンポ/尺/相槌を制御 + 軽量誤り検出を追加したディレクター（簡素版）。"""

    def __init__(
        self,
        ollama_client=None,
        model_name: str = "qwen2.5:7b",
        use_mcp: bool = True,
        local_model: Optional[str] = None,
    ):
        # LLM/検索
        self.client = ollama_client
        self.model_name = model_name
        self.use_mcp_search = bool(use_mcp)
        self.mcp_adapter = MCPWebSearchAdapter(language="ja") if self.use_mcp_search else None
        self.entity_llm_model = local_model or self.model_name

        # 会話状態
        self.temperature = 0.2
        self.current_phase = "flow"  # flow→wrap
        self.turn_counter = 0
        self.participants: List[str] = []  # [nameA, nameB]
        self.session_theme: Optional[str] = None

        # メトリクス
        self.metrics: Dict[str, Any] = {
            "avg_chars_last3": 0.0,
            "question_ratio": 0.0,
            "consecutive_by_last": 0,
            "last_speaker": None,
            "last_speaker_label": None,
        }

    # ===== 互換補助: UI/Manager が参照するユーティリティ =====
    def generate_length_instruction(self, guide: str = "簡潔") -> str:
        g = (guide or "").strip()
        if g in ("現状維持", "維持"):
            return "応答は現状の長さで問題ありません。箇条書きや見出しは使わず、自然な会話文で。"
        if g in ("長め", "やや長め"):
            return "やや長めに、200〜300文字・最大4文で。前置きは短く、最後に1つだけ短い問いを添えて。"
        if g in ("標準", "普通"):
            return "標準的な長さで、120〜160文字・最大3文。箇条書きは禁止。"
        # 既定: 簡潔
        return "簡潔に、80〜120文字・最大2文。要点だけを述べ、過度な称賛や長い前置きは避ける。"

    def should_end_dialogue(self, dialogue_history: List[Dict[str, Any]], turn_count: int) -> Tuple[bool, str]:
        try:
            if int(turn_count) >= 20:
                return True, "max_turns_reached"
        except Exception:
            pass
        try:
            stats = self._analyze(dialogue_history)
            lo, hi = QUESTION_RATIO_TARGET
            if self.current_phase == "wrap" and lo <= float(stats.get("question_ratio", 0.0)) <= hi and turn_count >= 16:
                return True, "wrap_phase_completed"
        except Exception:
            pass
        return False, "continue"

    def build_history_metrics_block(self, recent_texts: List[str]) -> str:
        """（簡素版）履歴から古典引用っぽさを超簡易カウントして返す（UI向け）。"""
        try:
            texts = [t for t in (recent_texts or []) if isinstance(t, str) and t.strip()]
            if not texts:
                return ""
            classic_refs = sum(len(re.findall(r"[「『][^」』]{4,30}[」』]", t)) for t in texts)
            return (
                "[history-metrics]\n"
                f"classic_refs = {int(classic_refs)}\n"
                "kyukokumei_refs = 0\n"
                "food_refs = 0\n"
                "[/history-metrics]"
            )
        except Exception:
            return ""

    def plan_next_turn(self, dialogue_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        stats = self._analyze(dialogue_context)
        speaker = self._pick_speaker(stats)
        speech_act = self._pick_speech_act(stats)
        max_chars = self._decide_max_chars(stats)
        aizuchi = self._decide_aizuchi(stats)
        closing_hint = self._decide_closing(stats)
        ban_list = [
            "praise",
            "long_intro",
            "list_format",
            "other_character_name",
            "speaker_label_prefix",
        ]
        return {
            "turn_style": {
                "speaker": speaker,
                "length": {"max_chars": max_chars, "max_sentences": MAX_SENTENCES},
                "preface": {
                    "aizuchi": aizuchi["on"],
                    "aizuchi_list": [aizuchi["word"]] if aizuchi["on"] else [],
                    "prob": aizuchi["prob"],
                },
                "speech_act": speech_act,
                "follow_up": "ask_feel" if speech_act == "ask" else "none",
                "ban": ban_list,
            },
            "cadence": {
                "avoid_consecutive_monologues": True,
                "enforce_question_ratio": list(QUESTION_RATIO_TARGET),
            },
            "closing_hint": closing_hint,
        }

    def update_phase(self, turn_count: int) -> None:
        self.current_phase = "wrap" if turn_count >= 16 else "flow"

    def generate_opening_instruction(self, theme: str, first_speaker_name: str) -> Dict[str, Any]:
        instruction = (
            f"テーマ『{theme}』について、挨拶は省き、自然な導入でやや詳しく語り始めてください。"
            f" 相手（{first_speaker_name}の相手）が反応しやすい、具体例や観点を一つ添え、2〜4文で。"
            " 箇条書きは使わず、会話文で。"
        )
        focus_points = [
            "やや長め (200-300文字)",
            "2〜4文で流れるように",
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

        def is_question(text: str) -> bool:
            return text.strip().endswith("？") or any(q in text for q in ["？", "?", "どう思う", "なぜ", "理由", "どこ", "いつ", "だれ", "何"])
        last10 = [m for m in dialogue_context if m.get("speaker") != "Director"][-10:]
        q_cnt = sum(1 for m in last10 if is_question(m.get("message", "")))
        question_ratio = (q_cnt/len(last10)) if last10 else 0.0

        # 参加者名A/B
        if dialogue_context:
            names: List[str] = []
            for m in dialogue_context:
                s = m.get("speaker")
                if s and s != "Director" and s not in names:
                    names.append(s)
                if len(names) >= 2:
                    break
            if names:
                if not self.participants:
                    self.participants = names[:2]
                else:
                    for n in names:
                        if n not in self.participants and len(self.participants) < 2:
                            self.participants.append(n)

        last_speaker = self.metrics["last_speaker"]
        current_last = last3[-1]["speaker"] if last3 else None
        consecutive = self.metrics["consecutive_by_last"] + 1 if current_last == last_speaker and current_last is not None else 1
        last_label = self._label_of(current_last) if current_last else None
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
        if stats["consecutive_by_last"] >= MAX_CONSECUTIVE_BY_SPEAKER and (stats.get("last_speaker_label") or stats.get("last_speaker")):
            return self._other(stats.get("last_speaker_label") or stats.get("last_speaker"))
        return self._other(stats.get("last_speaker_label") or stats.get("last_speaker")) if stats.get("last_speaker") else "A"

    def _pick_speech_act(self, stats: Dict[str, Any]) -> str:
        lo, hi = QUESTION_RATIO_TARGET
        if stats["question_ratio"] < lo:
            return "ask"
        if stats["avg_chars_last3"] > 110:
            return random.choice(["reflect", "agree_short", "handoff"])
        return random.choice(["answer", "reflect", "agree_short", "disagree_short"])

    def _decide_max_chars(self, stats: Dict[str, Any]) -> int:
        base_prob = 0.22
        if self.turn_counter <= 2:
            base_prob = 0.55
        long_turn = random.random() < base_prob
        if long_turn and stats["question_ratio"] <= QUESTION_RATIO_TARGET[1]:
            return random.choice([130, 140, 150, 160])
        if stats["avg_chars_last3"] > 120:
            return 85
        if stats["avg_chars_last3"] < 45:
            return 120
        return 95

    def _decide_aizuchi(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        on = random.random() < AIZUCHI_PROB_DEFAULT
        return {"on": on, "word": random.choice(AIZUCHI_POOL), "prob": AIZUCHI_PROB_DEFAULT}

    def _decide_closing(self, stats: Dict[str, Any]) -> str:
        if self.turn_counter >= 14 and QUESTION_RATIO_TARGET[0] <= stats["question_ratio"] <= QUESTION_RATIO_TARGET[1]:
            return "小まとめして交代"
        if self.turn_counter >= 18:
            return "締めに向かう"
        return "続ける"

    # ===== evaluate_dialogue（軽量ファクトチェック + リズム制御） =====
    async def evaluate_dialogue(self, dialogue_context: List[Dict[str, Any]], mode: Optional[str] = None) -> Dict[str, Any]:
        # カウンタ更新
        try:
            self.turn_counter = int(self.turn_counter) + 1
        except Exception:
            self.turn_counter = 1

        # 既定リズム指示（常に用意しておく）
        cmd_rhythm = self.plan_next_turn(dialogue_context)

        # 直近の非Director発話とその一つ前
        last_entry = None
        for m in reversed(dialogue_context):
            if m.get("speaker") and m.get("speaker") != "Director":
                last_entry = m
                break

        debug_info: Dict[str, Any] = {"verifications": []}

        # 軽量ファクトチェック
        if last_entry and isinstance(last_entry.get("message"), str):
            offender_label = self._label_of(last_entry.get("speaker") or "")
            text0 = last_entry.get("message", "")
            claims = self._extract_atomic_claims(text0)
            if claims:
                findings = self._verify_claims_light(claims)
                debug_info["verifications"] = findings
                if any(f.get("status") in ("false", "unknown") for f in findings):
                    dirx = self._build_factcheck_directives(findings, offender_label)
                    if dirx:
                        plan = self._build_pushback_plan(dirx["to"], "断定は避けて、条件と指標を指定して？")
                        plan = self._scale_plan_length(plan, factor=2, cap=200)
                        return {
                            "intervention_needed": True,
                            "reason": "light_factcheck",
                            "intervention_type": "challenge_and_verify",
                            "message": json.dumps(plan, ensure_ascii=False),
                            "response_length_guide": "簡潔",
                            "confidence": 0.78,
                            "director_debug": debug_info,
                            "review_directives": {
                                "required_actions": ["ask_for_task_metric_time_locale"],
                                "avoid": ["praise", "list_format"],
                                "tone_hint": "端的/助言調",
                                "ttl_turns": 2,
                            },
                        }

        # フォールバック: リズム/長さ/話法ガイド
        return {
            "intervention_needed": True,
            "reason": "rhythm_control",
            "intervention_type": "length_tempo_speech_act",
            "message": json.dumps(cmd_rhythm, ensure_ascii=False),
            "response_length_guide": "簡潔",
            "confidence": 0.8,
            "director_debug": debug_info,
        }

    # ===== 軽量ファクトチェック補助 =====
    def _other(self, s: Optional[str]) -> str:
        if s is None:
            return "A"
        label = s if s in ("A", "B") else self._label_of(s)
        return "B" if label == "A" else "A"

    def _label_of(self, name: str) -> str:
        if not name:
            return "A"
        if self.participants:
            if len(self.participants) >= 1 and name == self.participants[0]:
                return "A"
            if len(self.participants) >= 2 and name == self.participants[1]:
                return "B"
        return "A"

    def _extract_atomic_claims(self, text: str) -> List[Dict[str, str]]:
        if not text or len(text) < 15:
            return []
        # まずは簡易ヒューリスティックで抽出
        sentences = re.split(r"[。.!?！？]\s*", text.strip())
        candidates: List[str] = []
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if re.search(r"(は|とは).*(必要|重要|可能|増加|減少|達成|成立|求め|要る)", s):
                candidates.append(s)
            elif len(s) >= 20 and ("である" in s or "だ。" in s or "とされる" in s):
                candidates.append(s)
            if len(candidates) >= 2:
                break
        out: List[Dict[str, str]] = [{"id": f"C{i+1}", "text": c, "type": "fact"} for i, c in enumerate(candidates)]
        # LLM が使えるなら上書きでより良い抽出を試みる（失敗しても黙って戻る）
        try:
            if self.client:
                system = (
                    "あなたはファクトチェッカー。意見/感想/比喩は除外し、検証可能な原子主張のみをJSONで返す。"
                )
                user = (
                    f"発話:\n{text}\n\nJSONのみ:\n"
                    "{\"claims\":[{\"id\":\"C1\",\"text\":\"...\",\"type\":\"fact\"}]}"
                )
                resp = self.client.chat(
                    model=self.entity_llm_model,
                    messages=[{"role":"system","content":system},{"role":"user","content":user}],
                    options={"temperature":0.1,"num_ctx":2048}, stream=False,
                )
                raw = (resp.get("message") or {}).get("content") or ""
                s = raw[raw.find("{"): raw.rfind("}")+1] if "{" in raw and "}" in raw else raw
                data = json.loads(s)
                arr = (data or {}).get("claims") or []
                new_out: List[Dict[str, str]] = []
                for i, c in enumerate(arr[:5], 1):
                    t = (c.get("text") or "").strip()
                    ty = (c.get("type") or "fact").lower()
                    if len(t) >= 8:
                        new_out.append({"id": f"C{i}", "text": t, "type": ty})
                if new_out:
                    out = new_out[:3]
        except Exception:
            pass
        return out[:3]

    def _verify_claims_light(self, claims: List[Dict[str,str]]) -> List[Dict[str,Any]]:
        if not claims:
            return []
        out: List[Dict[str, Any]] = []
        for c in claims:
            q1, q2 = self._build_queries_for_claim(c)
            hits: List[Dict[str, Any]] = []
            try:
                if self.use_mcp_search and self.mcp_adapter:
                    if q1:
                        hits += self.mcp_adapter.search_snippets(q1, limit=3) or []
                    if q2 and q2 != q1:
                        hits += self.mcp_adapter.search_snippets(q2, limit=3) or []
            except Exception:
                pass
            hits = self._dedup_hits(hits)[:3]
            verdict, reason = self._rule_judge_claim(c, hits)
            out.append({
                "id": c["id"], "claim": c["text"], "type": c.get("type", "fact"),
                "status": verdict, "reason": reason, "evidence": hits
            })
        return out

    def _build_queries_for_claim(self, c: Dict[str,str]) -> Tuple[Optional[str], Optional[str]]:
        t = c.get("text"," ").strip()
        ty = c.get("type","fact").lower()
        if not t:
            return (None, None)
        if ty == "statistic":
            return (t + " site:go.jp", t + " 統計")
        if ty == "citation":
            return (f'"{t}" 出典', f'"{t}" 引用')
        return (t, None)

    def _dedup_hits(self, arr: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        seen = set(); out: List[Dict[str, Any]] = []
        for h in arr or []:
            u = (h.get("url") or "").split("#")[0]
            if u and u not in seen:
                seen.add(u); out.append(h)
        return out

    def _rule_judge_claim(self, c: Dict[str, Any], hits: List[Dict[str, Any]]) -> Tuple[str, str]:
        txt = c.get("text","")
        # 比較・断定一般論は不足スロットで unknown
        if any(k in txt for k in ["より", "優れて", "得意", "最も", "No.1", "世界一"]):
            return ("unknown", "タスク/指標/期間/条件が未指定")
        if not hits:
            return ("unknown", "一次資料が不足")
        if any(any(d in (h.get("url") or "") for d in [".go.jp", "data.gov", "who.int"]) for h in hits):
            return ("true", "公的ソースの裏付けあり（簡易判定）")
        return ("unknown", "決定的証拠が不足（保守的判定）")

    def _build_factcheck_directives(self, findings: List[Dict[str,Any]], offender_label: str) -> Optional[Dict[str,Any]]:
        focus_ids = [f["id"] for f in findings if f.get("status") in ("false","unknown")][:3]
        if not focus_ids:
            return None
        return {
            "required_actions": ["one_concise_question", "avoid_list", "avoid_praise"],
            "ensure": ["point_out_missing_slots(Task,Metric,Time,Locale)"],
            "avoid": ["long_intro"],
            "tone_hint": "端的/助言調",
            "focus_ids": focus_ids,
            "to": self._other(offender_label),
        }

    # ===== 既存の各種ユーティリティ =====
    def get_intervention_stats(self) -> Dict[str, Any]:
        return {}

    def get_resource_usage(self) -> Dict[str, Any]:
        return {
            "performance": {"avg_response_time": None},
            "resources": {"mcp_calls": "enabled" if self.use_mcp_search else "disabled"},
            "costs": {},
        }

    def _extract_history_metrics(self, dialogue_context: List[Dict[str, Any]]) -> Optional[Dict[str, int]]:
        return None

    def _build_pushback_plan(self, target_label: str, pushback_text: str) -> Dict[str, Any]:
        try:
            target = target_label if target_label in ("A", "B") else self._label_of(target_label)
        except Exception:
            target = "A"
        max_chars = 120
        plan = {
            "turn_style": {
                "speaker": target,
                "length": {"max_chars": max_chars, "max_sentences": 2},
                "preface": {"aizuchi": True, "aizuchi_list": [self.auto_repair(pushback_text, max_chars=70)], "prob": 1.0},
                "speech_act": "ask",
                "follow_up": "ask_feel",
                "ban": ["praise", "list_format", "long_intro", "speaker_label_prefix"],
            },
            "cadence": {
                "avoid_consecutive_monologues": True,
                "enforce_question_ratio": list(QUESTION_RATIO_TARGET),
            },
        }
        return plan

    def _scale_plan_length(self, plan: Dict[str, Any], factor: float = 1.5, cap: int = TURN_CHAR_MAX) -> Dict[str, Any]:
        try:
            ts = plan.get("turn_style", {})
            ln = ts.get("length", {})
            cur = int(ln.get("max_chars", TURN_CHAR_MEDIAN_TARGET))
            new_v = int(cur * float(factor))
            ln["max_chars"] = min(max(40, new_v), int(cap))
            ts["length"] = ln
            plan["turn_style"] = ts
            return plan
        except Exception:
            return plan

    def _build_refocus_plan(self, theme: str, target_label: str = "B") -> Dict[str, Any]:
        try:
            target = target_label if target_label in ("A", "B") else self._label_of(target_label)
        except Exception:
            target = "B"
        hint = f"ごめん、少しだけ整理。テーマ『{theme}』に戻して一言で要点を教えて？"
        plan = {
            "turn_style": {
                "speaker": target,
                "length": {"max_chars": 110, "max_sentences": 2},
                "preface": {"aizuchi": True, "aizuchi_list": [self.auto_repair(hint, max_chars=80)], "prob": 1.0},
                "speech_act": "ask",
                "follow_up": "none",
                "ban": ["praise", "list_format", "long_intro", "speaker_label_prefix"],
            },
            "cadence": {
                "avoid_consecutive_monologues": True,
                "enforce_question_ratio": list(QUESTION_RATIO_TARGET),
            },
        }
        return plan

    @staticmethod
    def judge_text(text: str, max_chars: int = TURN_CHAR_MAX, max_sentences: int = MAX_SENTENCES) -> Dict[str, Any]:
        violations = []
        if len(text) > max_chars:
            violations.append("too_long")
        sentences = re.split(r"[。.!?！？]+", text.strip())
        sentences = [s for s in sentences if s]
        if len(sentences) > max_sentences:
            violations.append("too_many_sentences")
        if LIST_MARKERS_PAT.search(text):
            violations.append("list_detected")
        if any(w in text for w in PRAISE_WORDS):
            violations.append("praise_used")
        if LONG_INTRO_PAT.search(text):
            violations.append("long_intro")
        return {"ok": not violations, "violations": violations}

    @staticmethod
    def auto_repair(text: str, max_chars: int = 90) -> str:
        text = LONG_INTRO_PAT.sub("", text)
        text = re.sub(r"^(?:[-*・]|\d+\.)\s*", "", text, flags=re.MULTILINE)
        for w in PRAISE_WORDS:
            text = text.replace(w, "(省略)")
        if len(text) > max_chars:
            text = text[:max_chars].rstrip("、, ")
        return text


# 互換エイリアス
AutonomousDirector = NaturalConversationDirector

# === 追加: プロンプト定義（クラス上部 or __init__直下でも可） ===
EXTRACT_SYSTEM = """あなたは辛口の査読者。比喩や価値判断を事実と混同しない。出力はJSONのみ。"""
EXTRACT_INSTR = """目的: 入力本文から最大8件の命題を抽出し、タイプ付けし、本文からの引用と文字オフセットを付けて返す。
タイプ: fact | causal | analogy | value | normative | prediction | definition | implicit
必須: 各命題は quote.text と quote.span.start/end(0-based,end非包含)を持つ。出力はJSONのみ、余談禁止。

出力スキーマ:
{"claims":[{"id":"C1","text":"<命題>","type":"fact","quote":{"text":"<引用<=120字>","span":{"start":0,"end":10}}}]}

本文:
{{TEXT}}"""

JUDGE_SYSTEM = """あなたは論証の査読者。命題の検証可能性を評価する。出力はJSONのみ。"""
JUDGE_INSTR = """目的: 与えられた claims を判定して返す。
判定: supported | contradicted | needs-evidence | non-falsifiable | unclear
必須: 各判定に reason(1文) を付ける。needs-evidence なら required_evidence(データ/文献の型)を列挙。suggestion は任意。

入出力:
入力: {"claims":[{"id":"C1","text":"...","type":"fact"}]}
出力:
{"findings":[{"claim_id":"C1","status":"needs-evidence","reason":"...", "required_evidence":["..."], "suggestion":"..."}],
 "summary":{"counts_by_status":{},"top_risks":[]}}"""

# === 追加: LLM JSON ヘルパ ===
def _llm_json(self, system: str, user: str) -> Optional[dict]:
    if not self.client:
        return None
    resp = self.client.chat(
        model=self.entity_llm_model,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        options={"temperature":0.1, "num_ctx": 4096},
        stream=False,
    )
    raw = (resp.get("message") or {}).get("content") or ""
    # ゆるいJSON抽出
    s = raw[raw.find("{") : raw.rfind("}")+1] if "{" in raw and "}" in raw else raw
    try:
        return json.loads(s)
    except Exception:
        return None

# === 置換: 原子主張抽出（LLM優先→既存ヒューリスティックにフォールバック） ===
def _extract_atomic_claims(self, text: str) -> List[Dict[str, str]]:
    if not text or len(text) < 15:
        return []
    out: List[Dict[str, str]] = []
    # 1) LLMで抽出
    try:
        payload = EXTRACT_INSTR.replace("{{TEXT}}", text)
        data = self._llm_json(EXTRACT_SYSTEM, payload)  # type: ignore[attr-defined]
        claims = (data or {}).get("claims") or []
        for i, c in enumerate(claims[:5], 1):
            t = (c.get("text") or "").strip()
            ty = (c.get("type") or "fact").lower()
            if len(t) >= 6:
                out.append({"id": f"C{i}", "text": t, "type": ty})
        if out:
            return out[:3]
    except Exception:
        pass

    # 2) フォールバック: 既存ヒューリスティック
    sentences = re.split(r"[。.!?！？]\s*", text.strip())
    candidates: List[str] = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if re.search(r"(は|とは).*(必要|重要|可能|増加|減少|達成|成立|求め|要る)", s):
            candidates.append(s)
        elif len(s) >= 20 and ("である" in s or "だ。" in s or "とされる" in s):
            candidates.append(s)
        if len(candidates) >= 2:
            break
    out = [{"id": f"C{i+1}", "text": c, "type": "fact"} for i, c in enumerate(candidates)]
    return out[:3]

# === 追加: 命題判定（LLM）。返り値は {claim_id: finding} マップ ===
def _judge_claims_llm(self, claims: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    if not claims or not self.client:
        return {}
    inp = {"claims": [{"id": c["id"], "text": c["text"], "type": c.get("type","fact")} for c in claims]}
    data = self._llm_json(JUDGE_SYSTEM, JUDGE_INSTR + "\n\n入力:\n" + json.dumps(inp, ensure_ascii=False))  # type: ignore[attr-defined]
    findings = (data or {}).get("findings") or []
    out = {}
    for f in findings:
        cid = f.get("claim_id")
        if cid:
            out[cid] = f
    return out

# === 置換: 軽量検証（LLM判定→既存ルール/検索の結果を統合） ===
def _verify_claims_light(self, claims: List[Dict[str,str]]) -> List[Dict[str,Any]]:
    if not claims:
        return []
    out: List[Dict[str, Any]] = []
    # まず LLM 判定を取得（失敗時は空マップ）
    judge_map = {}
    try:
        judge_map = self._judge_claims_llm(claims)
    except Exception:
        judge_map = {}

    def map_status(s: str) -> str:
        s = (s or "").lower()
        if s in ("supported",):
            return "true"
        if s in ("contradicted",):
            return "false"
        # needs-evidence / non-falsifiable / unclear は unknown 扱い（押し返し対象）
        return "unknown"

    for c in claims:
        q1, q2 = self._build_queries_for_claim(c)
        hits: List[Dict[str, Any]] = []
        try:
            if self.use_mcp_search and self.mcp_adapter:
                if q1: hits += self.mcp_adapter.search_snippets(q1, limit=3) or []
                if q2 and q2 != q1: hits += self.mcp_adapter.search_snippets(q2, limit=3) or []
        except Exception:
            pass
        hits = self._dedup_hits(hits)[:3]

        # 既存の簡易ルール
        verdict, reason = self._rule_judge_claim(c, hits)

        # LLM判定があれば優先しつつ、証拠ヒットの有無も加味
        f = judge_map.get(c["id"], {})
        if f:
            verdict = map_status(f.get("status"))
            # 公的ソースに当たっていれば supported を true で上書き（保守的に強化）
            if verdict != "true" and any(any(d in (h.get("url") or "") for d in [".go.jp", "data.gov", "who.int"]) for h in hits):
                verdict = "true"
            reason = f.get("reason") or reason

        out.append({
            "id": c["id"],
            "claim": c["text"],
            "type": c.get("type", "fact"),
            "status": verdict,          # "true" | "false" | "unknown"（既存ロジック互換）
            "reason": reason,
            "suggestion": (f.get("suggestion") if f else None),
            "required_evidence": (f.get("required_evidence") if f else []),
            "evidence": hits
        })
    return out
