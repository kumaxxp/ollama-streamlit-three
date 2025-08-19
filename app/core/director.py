# natural_conversation_director.py

"""
Director AI (v3) — Natural Conversation Director + Light Factcheck
目的: 議論の手触り（リズム/尺/相槌）を最優先しつつ、軽量の誤り検出パイプ
      （抽出→Webスニペット検証→指示）を最小コストで挿入する。
- 介入はテンポ/話法の指定(JSON)を基本に、必要時のみ短い“確認/訂正要求”を生成。
- 既存AutonomousDirector互換。
"""

from __future__ import annotations
import json
import logging
import os
import random
import re
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from .search_adapter import MCPWebSearchAdapter

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
TURN_CHAR_MIN = 30
TURN_CHAR_MAX = 160
TURN_CHAR_MEDIAN_TARGET = 85
MAX_SENTENCES = 3
QUESTION_RATIO_TARGET = (0.30, 0.55)
AIZUCHI_PROB_DEFAULT = 0.35
MAX_CONSECUTIVE_BY_SPEAKER = 2

# 一般語の簡易ストップワード（誤検出抑制。必要に応じて拡張）
STOP_ENTITY_COMMON = set([
    "負荷", "時間", "今日", "明日", "昨日", "世界", "社会", "問題", "方法", "議論",
    "情報", "研究", "教育", "経済", "技術", "文化", "日本", "人間", "私", "あなた",
    "AI", "IT", "SNS", "CPU", "GPU", "OS", "PC", "NG", "OK",
    "会話", "議題", "話題", "テーマ", "意見", "考え", "気持ち", "全部", "本当", "自然", "数字", "欲求",
    "結局", "古都", "本物", "再現", "データ", "解釈",
])

class RateLimitManager:
    """Gemini API レート制限の簡易管理（オフラインでも動作可能なスタブ）。"""

    def __init__(self):
        self.minute_limit = 15
        self.daily_limit = 1500
        self.minute_used = 0
        self.daily_used = 0

    def get_current_usage(self) -> Dict[str, Any]:
        return {
            "minute": {"used": self.minute_used, "limit": self.minute_limit, "remaining": max(0, self.minute_limit - self.minute_used)},
            "daily": {"used": self.daily_used, "limit": self.daily_limit, "remaining": max(0, self.daily_limit - self.daily_used)},
            "reset_time": None,
        }

    def should_use_cloud(self) -> Tuple[bool, str]:
        usage = self.get_current_usage()
        daily_percent = usage["daily"]["remaining"] / usage["daily"]["limit"] if usage["daily"]["limit"] else 0.0
        if daily_percent < 0.2:
            return False, "daily_limit_warning"
        if usage["minute"]["remaining"] < 2:
            return False, "minute_limit"
        return True, "ok"

    def note_call(self, n: int = 1):
        self.minute_used += n
        self.daily_used += n


class CostOptimizer:
    """API 使用量の簡易最適化（時間帯でバッチサイズを調整）。"""

    def __init__(self, daily_budget: int = 1500):
        self.daily_budget = daily_budget
        self.hourly_target = max(1, daily_budget // 24)

    def get_optimal_batch_size(self, current_hour: Optional[int] = None) -> int:
        from datetime import datetime as _dt
        h = _dt.now().hour if current_hour is None else current_hour
        if 0 <= h < 6:
            return 1
        elif 9 <= h < 18:
            return 5
        return 3


class GeminiErrorDetector:
    """Gemini 2.0 Flash によるエラー検出（APIキーがある場合のみ動作）。"""

    def __init__(self, api_key: Optional[str]):
        self.enabled = False
        self.batch_size = 3
        self.detection_queue: List[Dict[str, Any]] = []
        self._model = None
        try:
            if api_key:
                import google.generativeai as genai  # type: ignore
                genai.configure(api_key=api_key)
                self._model = genai.GenerativeModel("gemini-2.0-flash-exp")
                self.enabled = True
        except Exception:
            self.enabled = False
            self._model = None

    def add_entries(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not entries:
            return []
        self.detection_queue.extend(entries)
        if len(self.detection_queue) < self.batch_size:
            return []
        batch = self.detection_queue[: self.batch_size]
        self.detection_queue = self.detection_queue[self.batch_size :]
        return self._process_batch(batch)

    def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            if not self.enabled or not self._model:
                return []
            prompt = (
                "以下の発話を同時に分析し、それぞれについて次を検出しJSONで返してください:\n"
                "1) 事実誤認 2) 論理矛盾 3) ハルシネーション 4) 固有名詞の誤用\n"
                "問題点の検出仕様はそのままに、各発話について発言者に向けた短い『ツッコミ/訂正要求』文も1つ生成してください。\n"
                "出力要件: JSONのみ。pushback は日本語で20-60文字、具体的で端的に。\n"
                "必ず次の形式で: {\"results\":[{\"id\":str,\"errors\":[{\"type\":str,\"detail\":str,\"entity\":str|null}],\"pushback\":str}]}\n\n"
                + json.dumps(batch, ensure_ascii=False)
            )
            resp = self._model.generate_content(prompt)
            text = getattr(resp, "text", None)
            if isinstance(text, str) and text.strip():
                return self._parse_batch_response(text)
            try:
                return self._parse_batch_response(str(resp))
            except Exception:
                return []
        except Exception:
            return []

    def _parse_batch_response(self, s: str) -> List[Dict[str, Any]]:
        try:
            ss = s.strip()
            start = ss.find("{")
            end = ss.rfind("}")
            if start >= 0 and end > start:
                ss = ss[start : end + 1]
            data = json.loads(ss)
            if isinstance(data, dict) and isinstance(data.get("results"), list):
                return data["results"]  # type: ignore[return-value]
        except Exception:
            return []
        return []


class NaturalConversationDirector:
    """会話のテンポ/尺/相槌を制御 + 軽量誤り検出を追加したディレクター。"""

    def __init__(
        self,
        ollama_client=None,
        model_name: str = "qwen2.5:7b",
        gemini_api_key: Optional[str] = None,
        use_mcp: bool = True,
        local_model: Optional[str] = None,
    ):
        # LLM
        self.client = ollama_client
        self.model_name = model_name
        self.temperature = 0.2
        self.current_phase = "flow"  # flow→wrap
        self.turn_counter = 0

        # 参加者名
        self.participants = []  # [nameA, nameB]

        # メトリクス
        self.metrics = {
            "avg_chars_last3": 0.0,
            "question_ratio": 0.0,
            "consecutive_by_last": 0,
            "last_speaker": None,
            "last_speaker_label": None,
        }

        # 監視/制限/費用最適化
        self.rate_limiter = RateLimitManager()
        self.cost_optimizer = CostOptimizer()
        self.gemini_detector = GeminiErrorDetector(gemini_api_key)

        # MCP/検索
        self.use_mcp_search = bool(use_mcp)
        self.mcp_adapter = MCPWebSearchAdapter(language="ja") if self.use_mcp_search else None

        # キャッシュ
        self.entity_cache = {}
        self.shared_knowledge_cache = {}

        # LLM抽出設定
        self.use_llm_entity_fallback = True
        self.use_heuristics = bool(int(os.getenv("DIRECTOR_USE_HEURISTICS", "0")))  # 0=OFF(既定)
        self.always_llm_entity_scan = bool(int(os.getenv("DIRECTOR_ALWAYS_LLM_ENTITY_SCAN", "1")))  # 1=ON(既定)
        self.entity_llm_model = local_model or self.model_name

        # LLM主導の異常検出
        self.use_llm_anomaly_detector = True
        self.max_anomaly_checks_per_turn = 4

        # Soft Ack
        self.soft_ack_enabled = True
        self.soft_ack_max_reminders = 2
        self.soft_ack_expire_window = 4
        self.pending_soft_ack = {}

        # 介入統計
        self._intervention_stats = {
            "entity_checks": 0,
            "entity_corrections": 0,
            "soft_ack_dispatched": 0,
            "gemini_batches": 0,
            "rhythm_guides": 0,
        }

        # エンティティ検出設定
        self.entity_detection_mode = "strict"  # strict | balanced
        self.max_entity_checks_per_turn = 2
        try:
            self.prefer_text_llm_output = bool(int(os.getenv("DIRECTOR_PREFER_TEXT_LLM_OUTPUT", "1")))
        except Exception:
            self.prefer_text_llm_output = True

        # リフォーカス
        self.session_theme = None
        self._eval_calls = 0
        self._last_refocus_eval = -999
        self._refocus_cooldown = 3  # 何回の分析間隔おきに許可するか

    def plan_next_turn(self, dialogue_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """リズム/長さ/話法のガイドを決める。"""
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

        cmd = {
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
        return cmd

    def update_phase(self, turn_count: int) -> None:
        """ターン数に応じて内部フェーズ更新。"""
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
        if current_last == last_speaker and current_last is not None:
            consecutive = self.metrics["consecutive_by_last"] + 1
        else:
            consecutive = 1
        last_label = None
        if current_last:
            last_label = self._label_of(current_last)
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

    # ===== evaluate_dialogue（軽量ファクトチェック挿入版） =====
    async def evaluate_dialogue(self, dialogue_context: List[Dict[str, Any]], mode: Optional[str] = None) -> Dict[str, Any]:
        """
        旧API互換 + 早期に『軽量ファクトチェック』を1ショット実行。
        優先度:
          0) 軽量ファクトチェック（抽出→MCPスニペ検証→短い確認依頼）
          1) 固有名詞チェックに基づく訂正促し
          2) soft-ack / リズム制御
        """
        # ★ 先頭で turn_counter を確実に進める（従来の不具合修正）
        try:
            self.turn_counter = int(self.turn_counter) + 1
        except Exception:
            self.turn_counter = 1

        # 通常のリズム計画（フォールバック）
        cmd_rhythm = self.plan_next_turn(dialogue_context)
        try:
            self._eval_calls += 1
        except Exception:
            self._eval_calls = 1

        debug_info: Dict[str, Any] = {
            "heuristic_entities": [],
            "llm_entities": [],
            "anomalies": [],
            "all_candidates": [],
            "verifications": [],
            "gemini": None,
            "holistic_text": None,
        }
        mode = (mode or "balanced").lower()

        metrics = self._extract_history_metrics(dialogue_context)
        if isinstance(metrics, dict):
            debug_info["history_metrics"] = metrics

        # 直近の非Director発話とその一つ前
        last_entry = None
        prev_entry = None
        for m in reversed(dialogue_context):
            if m.get("speaker") and m.get("speaker") != "Director":
                if last_entry is None:
                    last_entry = m
                else:
                    prev_entry = m
                    break

        _ = self._analyze(dialogue_context)

        # オフトピック検知
        try:
            theme = self.session_theme
            offtopic, reason = self._detect_offtopic(dialogue_context, theme)
        except Exception:
            offtopic, reason = (False, None)
        if offtopic and (self._eval_calls - self._last_refocus_eval >= self._refocus_cooldown):
            target_label = "B"
            plan = self._build_refocus_plan(theme or "この対話のテーマ", target_label)
            self._last_refocus_eval = self._eval_calls
            debug_info["offtopic"] = {"detected": True, "reason": reason or "theme_mismatch"}
            return {
                "intervention_needed": True,
                "reason": "offtopic_refocus",
                "intervention_type": "refocus_or_reset",
                "message": json.dumps(plan, ensure_ascii=False),
                "response_length_guide": "簡潔",
                "confidence": 0.75,
                "director_debug": debug_info,
            }

        # === ★ ここから挿入: 軽量ファクトチェック（抽出→検証→指示） ===
        if last_entry and isinstance(last_entry.get("message"), str):
            offender_disp = last_entry.get("speaker") or ""
            offender_label = self._label_of(offender_disp)
            text0 = last_entry.get("message", "")

            # A) 原子主張抽出
            claims = self._extract_atomic_claims(text0)
            if claims:
                # B) 軽量検証（MCPスニペット）
                findings = self._verify_claims_light(claims)
                debug_info["light_factcheck"] = {"claims": claims, "findings": findings}
                # C) 問題があれば短い指摘を先に返す（テンポ優先で1発のみ）
                if any(f["status"] in ("false", "unknown") for f in findings):
                    dirx = self._build_factcheck_directives(findings, offender_label)
                    if dirx:
                        # pushback テキストは短めに（既存のビルダーを再利用）
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
                                "avoid": ["praise","list_format"],
                                "tone_hint": "端的/助言調",
                                "ttl_turns": 2
                            },
                        }

        # （この先は従来のホリスティック/エンティティ/リズム制御へ続く）
        # 0) 文章ベースのホリスティック評価（既存ロジック）
        if last_entry and isinstance(last_entry.get("message"), str) and getattr(self, "prefer_text_llm_output", False):
            def _seems_correction(text: str) -> bool:
                try:
                    import re as _re
                    t = (text or "").lower()
                    patterns = [r"違|ちが|誤|間違|おかし|本当|ほんとう|根拠|出典|ソース|引用|どこ|ではなく|じゃない|嘘|デマ|検証|確か|wiki|wikipedia|参考|典拠|証拠"]
                    if _re.search("|".join(patterns), t):
                        return True
                    if "?" in text or "？" in text:
                        return True
                except Exception:
                    pass
                return False

            last_disp = last_entry.get("speaker") or ""
            last_text = last_entry.get("message", "")
            critic_mode = _seems_correction(last_text)
            if critic_mode and prev_entry and prev_entry.get("speaker"):
                offender_disp = prev_entry.get("speaker")
                offender_label = self._label_of(offender_disp)
                target_label = self._label_of(last_disp)
            else:
                offender_disp = last_disp
                offender_label = self._label_of(offender_disp)
                target_label = self._other(offender_label)
            text0 = last_text
            try:
                htx = self._llm_holistic_review_text(text0, dialogue_context)
            except Exception:
                htx = None
            if isinstance(htx, dict):
                debug_info["holistic_text"] = htx.get("full_text")
                if bool(htx.get("risk")):
                    try:
                        queries = htx.get("queries") or []
                        if self.use_mcp_search and self.mcp_adapter and isinstance(queries, list) and queries:
                            q0 = str(queries[0])
                            v, u, ex = self.mcp_adapter.verify_entity_detail(q0, "ENTITY")
                            debug_info["research"] = [{
                                "query": q0,
                                "verdict": v,
                                "evidence": u,
                                "evidence_text": ex,
                            }]
                            try:
                                lat, lon, in_jp = self.mcp_adapter.get_coordinates(q0)
                                if lat is not None and lon is not None:
                                    debug_info.setdefault("geo", {})[q0] = {
                                        "lat": lat, "lon": lon, "in_japan": in_jp
                                    }
                            except Exception:
                                pass
                    except Exception:
                        pass
                    try:
                        works = self._detect_classics_and_works(text0) or []
                        try:
                            rtxt = (htx.get("full_text") or "") if isinstance(htx, dict) else ""
                            if rtxt:
                                import re as _re
                                for m in _re.finditer(r"[「『]([^」』]{2,20})[」』]", rtxt):
                                    t = (m.group(1) or "").strip()
                                    if t and t not in works:
                                        works.append(t)
                        except Exception:
                            pass
                    except Exception:
                        works = []
                    try:
                        wiki_snippets = []
                        if self.use_mcp_search and self.mcp_adapter:
                            for w in works[:3]:
                                snips = self.mcp_adapter.search_snippets(w, limit=2) or []
                                for s in snips:
                                    wiki_snippets.append({"query": w, "title": s.get("title"), "url": s.get("url"), "excerpt": s.get("excerpt")})
                            q_extra = None
                            try:
                                qlist = htx.get("queries") or []
                                if isinstance(qlist, list) and qlist:
                                    q_extra = str(qlist[0])
                            except Exception:
                                q_extra = None
                            if q_extra and not works:
                                snips2 = self.mcp_adapter.search_snippets(q_extra, limit=2) or []
                                for s in snips2:
                                    wiki_snippets.append({"query": q_extra, "title": s.get("title"), "url": s.get("url"), "excerpt": s.get("excerpt")})
                        if wiki_snippets:
                            debug_info["wiki_snippets"] = wiki_snippets[:4]
                    except Exception:
                        pass
                    review_directives = {
                        "required_actions": [],
                        "ensure": ["one_concise_question"],
                        "avoid": ["praise", "list_format"],
                        "tone_hint": "端的/助言調",
                        "ttl_turns": 2,
                    }
                    try:
                        cref = int((metrics or {}).get("classic_refs", 0))
                        if cref >= 12:
                            review_directives["required_actions"].append("ban_classic_quotes")
                            if "overuse_classics" not in review_directives["avoid"]:
                                review_directives["avoid"].append("overuse_classics")
                    except Exception:
                        pass
                    if isinstance(debug_info.get("research"), list) and debug_info["research"]:
                        review_directives["required_actions"].append("cite_one_url_if_available")
                    geo = debug_info.get("geo") or {}
                    if isinstance(geo, dict) and any((isinstance(g, dict) and g.get("in_japan") is False) for g in geo.values()):
                        review_directives["required_actions"].append("geo_nudge")
                    if works:
                        review_directives["required_actions"].append("summarize_work")
                        review_directives["avoid"].append("overuse_classics")
                        debug_info["works_detected"] = []
                        verified_found = False
                        for w in works[:3]:
                            try:
                                row = self._summarize_or_validate_work(w)
                            except Exception:
                                row = {"title": w, "verdict": "AMBIGUOUS", "url": None, "summary": None}
                            debug_info["works_detected"].append(row)
                            if row.get("verdict") in ("NG", "AMBIGUOUS"):
                                if "offer_correction" not in review_directives["required_actions"]:
                                    review_directives["required_actions"].append("offer_correction")
                            elif row.get("verdict") == "VERIFIED":
                                verified_found = True
                        if verified_found:
                            if "check_verified_relevance" not in review_directives["required_actions"]:
                                review_directives["required_actions"].append("check_verified_relevance")
                            review_directives["tone_hint"] = review_directives.get("tone_hint") or "端的/少し強めに"
                    plan = self._build_holistic_intervention_text(target_label, htx)
                    try:
                        if critic_mode and isinstance(plan, dict):
                            ts = plan.get("turn_style", {})
                            ts["speech_act"] = "ask"
                            pre = ts.get("preface", {}) or {}
                            pre["aizuchi"] = True
                            pre["aizuchi_list"] = ["ちょっと確認だけど…"]
                            ts["preface"] = pre
                            plan["turn_style"] = ts
                    except Exception:
                        pass
                    try:
                        wiki_snips = debug_info.get("wiki_snippets") or []
                        if wiki_snips:
                            second = self._llm_holistic_review_with_evidence(text0, wiki_snips[:3])
                            if isinstance(second, dict):
                                pb = (second.get("pushback") or "").strip()
                                if pb:
                                    try:
                                        plan_ts = plan.get("turn_style", {})
                                        pre = plan_ts.get("preface", {})
                                        pre["aizuchi_list"] = [self.auto_repair(pb, max_chars=70)]
                                        pre["aizuchi"] = True
                                        plan_ts["preface"] = pre
                                        plan["turn_style"] = plan_ts
                                    except Exception:
                                        pass
                                ad = second.get("agent_directives") or {}
                                if isinstance(ad, dict):
                                    for key in ("required_actions", "ensure", "avoid"):
                                        if isinstance(ad.get(key), list):
                                            review_directives.setdefault(key, [])
                                            for v in ad[key]:
                                                if v not in review_directives[key]:
                                                    review_directives[key].append(v)
                                    if isinstance(ad.get("tone_hint"), str):
                                        review_directives["tone_hint"] = ad.get("tone_hint")
                    except Exception:
                        pass
                    try:
                        has_materials = bool(debug_info.get("research") or debug_info.get("wiki_snippets") or debug_info.get("works_detected"))
                        if has_materials:
                            plan = self._scale_plan_length(plan, factor=3, cap=260)
                    except Exception:
                        pass
                    if self.soft_ack_enabled:
                        self._schedule_soft_ack(offender_label)
                        self._intervention_stats["soft_ack_dispatched"] += 1
                    return {
                        "intervention_needed": True,
                        "reason": "holistic_text_review",
                        "intervention_type": "challenge_and_verify",
                        "message": json.dumps(plan, ensure_ascii=False),
                        "response_length_guide": "簡潔",
                        "confidence": 0.8,
                        "director_debug": debug_info,
                        "review_directives": review_directives,
                    }

            try:
                works = self._detect_classics_and_works(text0) or []
                try:
                    rtxt = (htx.get("full_text") or "") if isinstance(htx, dict) else ""
                    if rtxt:
                        import re as _re
                        for m in _re.finditer(r"[「『]([^」』]{2,20})[」』]", rtxt):
                            t = (m.group(1) or "").strip()
                            if t and t not in works:
                                works.append(t)
                except Exception:
                    pass
            except Exception:
                works = []
            if works and self.use_mcp_search and self.mcp_adapter:
                try:
                    debug_info.setdefault("works_detected", [])
                    try:
                        wiki_snips = []
                        for w in works[:3]:
                            sn = self.mcp_adapter.search_snippets(w, limit=2) or []
                            for s in sn:
                                wiki_snips.append({"query": w, "title": s.get("title"), "url": s.get("url"), "excerpt": s.get("excerpt")})
                        if wiki_snips:
                            debug_info["wiki_snippets"] = wiki_snips[:4]
                    except Exception:
                        pass
                    flagged = False
                    verified_found = False
                    for w in works[:3]:
                        row = self._summarize_or_validate_work(w)
                        debug_info["works_detected"].append(row)
                        if row.get("verdict") in ("NG", "AMBIGUOUS"):
                            flagged = True
                        elif row.get("verdict") == "VERIFIED":
                            verified_found = True
                    if flagged:
                        if critic_mode and last_entry and last_entry.get("speaker"):
                            target_label = self._label_of(last_entry.get("speaker"))
                        else:
                            target_label = self._other(self._label_of(offender_disp))
                        review_directives = {
                            "required_actions": ["offer_correction", "one_concise_question"],
                            "avoid": ["praise", "list_format", "overuse_classics"],
                            "tone_hint": "端的/助言調",
                            "ttl_turns": 2,
                        }
                        pb = "その引用は出典あいまい。どの作品/章？"
                        plan = self._build_pushback_plan(target_label, pb)
                        try:
                            if critic_mode and isinstance(plan, dict):
                                ts = plan.get("turn_style", {})
                                ts["speech_act"] = "ask"
                                pre = ts.get("preface", {}) or {}
                                pre["aizuchi"] = True
                                pre["aizuchi_list"] = ["ごめん、確認なんだけど…"]
                                ts["preface"] = pre
                                plan["turn_style"] = ts
                        except Exception:
                            pass
                        plan = self._scale_plan_length(plan, factor=3, cap=260)
                        if self.soft_ack_enabled:
                            self._schedule_soft_ack(self._label_of(offender_disp))
                            self._intervention_stats["soft_ack_dispatched"] += 1
                        return {
                            "intervention_needed": True,
                            "reason": "quoted_phrase_unverified",
                            "intervention_type": "entity_correction",
                            "message": json.dumps(plan, ensure_ascii=False),
                            "response_length_guide": "簡潔",
                            "confidence": 0.75,
                            "director_debug": debug_info,
                            "review_directives": review_directives,
                        }
                except Exception:
                    pass
                self._intervention_stats["rhythm_guides"] += 1
                review_directives = {}
                try:
                    cref = int((metrics or {}).get("classic_refs", 0))
                    req: List[str] = []
                    ensure: List[str] = []
                    avoid: List[str] = ["praise", "list_format", "overuse_classics"]
                    if verified_found:
                        req.append("check_verified_relevance")
                    if cref >= 12:
                        req.append("ban_classic_quotes")
                        ensure.append("one_concise_question")
                    if req or ensure or avoid:
                        review_directives = {
                            "required_actions": req,
                            "ensure": ensure,
                            "avoid": avoid,
                            "tone_hint": "端的/助言調",
                            "ttl_turns": 2,
                            "proposed_pushback_verified": "その引用、今の話題と関係ある？要点だけで話そう。",
                        }
                except Exception:
                    review_directives = None
                return {
                    "intervention_needed": True,
                    "reason": "rhythm_control",
                    "intervention_type": "length_tempo_speech_act",
                    "message": json.dumps(cmd_rhythm, ensure_ascii=False),
                    "response_length_guide": "簡潔",
                    "confidence": 0.8,
                    "director_debug": debug_info,
                    **({"review_directives": review_directives} if review_directives else {}),
                }

        # 1) 固有名詞チェック（既存ロジック）…（以降はオリジナルのまま）
        # ・・・（中略：元の entity / anomaly / gemini 部分はそのまま）・・・

        # 2) Soft Ack
        if self.soft_ack_enabled and self.pending_soft_ack:
            now_turn = self.turn_counter
            for label, st in list(self.pending_soft_ack.items()):
                remaining = st.get("remaining", 0)
                expire = st.get("expire_turn", 0)
                if remaining > 0 and now_turn <= expire:
                    plan = self._build_soft_ack_plan(label)
                    st["remaining"] = remaining - 1
                    if st["remaining"] <= 0:
                        self.pending_soft_ack.pop(label, None)
                    else:
                        self.pending_soft_ack[label] = st
                    return {
                        "intervention_needed": True,
                        "reason": "soft_ack_tone",
                        "intervention_type": "tone_guidance_soft_ack",
                        "message": json.dumps(plan, ensure_ascii=False),
                        "response_length_guide": "簡潔",
                        "confidence": 0.8,
                        "director_debug": debug_info,
                    }
                if now_turn > expire:
                    self.pending_soft_ack.pop(label, None)

        # 3) フォールバック: リズム/長さ/話法ガイド
        self._intervention_stats["rhythm_guides"] += 1
        return {
            "intervention_needed": True,
            "reason": "rhythm_control",
            "intervention_type": "length_tempo_speech_act",
            "message": json.dumps(cmd_rhythm, ensure_ascii=False),
            "response_length_guide": "簡潔",
            "confidence": 0.8,
            "director_debug": debug_info,
        }

    # ===== 以降: 既存メソッド（verify_entityなど）＋ 新規の軽量ファクトチェック補助 =====

    def _detect_offtopic(self, dialogue_context: List[Dict[str, Any]], theme: Optional[str]) -> Tuple[bool, Optional[str]]:
        # （原実装のまま）
        try:
            import re as _re
            def key_tokens(s: str) -> List[str]:
                if not s:
                    return []
                t = _re.sub(r"[\s、。.,!！?？\-:：/（）()\[\]『』「」]", " ", s)
                toks = [w for w in t.split() if len(w) >= 2]
                stop = {"こと","もの","それ","これ","ため","よう","とか","から","ので","です","ます","する","ある","いる","的","場合","時","思う","感じ"}
                return [w for w in toks if w not in stop]
            last_msgs = [m.get("message", "") for m in dialogue_context if isinstance(m, dict) and m.get("speaker") != "Director"][-2:]
            if not last_msgs:
                return (False, None)
            theme_keys = set(key_tokens(theme or ""))
            if not theme_keys:
                return (False, None)
            recent_keys = set(key_tokens(" ".join(last_msgs)))
            overlap = theme_keys.intersection(recent_keys)
            length_ok = sum(len(x) for x in last_msgs) >= 30
            if length_ok and len(overlap) == 0:
                return (True, "no_theme_overlap")
        except Exception:
            return (False, None)
        return (False, None)

    def _other(self, s: Optional[str]) -> str:
        if s is None:
            return "A"
        label = s
        if s not in ("A", "B"):
            label = self._label_of(s)
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

    # ・・・（中略：既存の _llm_holistic_review_text / pushback builders / entity抽出 等はそのまま）・・・

    # ===== 軽量ファクトチェック：追加メソッド =====

    def _extract_atomic_claims(self, text: str) -> List[Dict[str, str]]:
        """
        直近発話から検証価値のある原子主張だけを抽出。
        出力: [{"id":"C1","text":"...","type":"fact|classical|statistic|citation|metaphor"}]
        """
        if not self.client or not text or len(text) < 20:
            return []
        system = ("あなたはファクトチェッカー。意見/感想/比喩は除外し、"
                  "検証可能な原子主張だけをJSONで抽出。比較主張は残す。")
        user = (
            f"発話:\n{text}\n\nJSONのみ:\n"
            "{\"claims\":[{\"id\":\"C1\",\"text\":\"...\",\"type\":\"fact\"}]}"
        )
        try:
            resp = self.client.chat(
                model=self.entity_llm_model,
                messages=[{"role":"system","content":system},{"role":"user","content":user}],
                options={"temperature":0.1,"num_ctx":2048}, stream=False,
            )
            raw = (resp.get("message") or {}).get("content") or ""
            s = raw[raw.find("{"): raw.rfind("}")+1] if "{" in raw and "}" in raw else raw
            data = json.loads(s)
            arr = (data or {}).get("claims") or []
            out = []
            for i, c in enumerate(arr[:5], 1):
                t = (c.get("text") or "").strip()
                ty = (c.get("type") or "fact").lower()
                if len(t) >= 8:
                    out.append({"id": f"C{i}", "text": t, "type": ty})
            return out[:3]
        except Exception:
            return []

    def _verify_claims_light(self, claims: List[Dict[str,str]]) -> List[Dict[str,Any]]:
        """
        各主張につき検索2パターン×3件、fetchは1件だけ（MVP）。
        verdict: true|false|unknown, reason, evidence[{title,url,snippet}]
        """
        if not claims:
            return []
        out = []
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
                "id": c["id"], "claim": c["text"], "type": c["type"],
                "status": verdict, "reason": reason, "evidence": hits
            })
        return out

    def _build_queries_for_claim(self, c: Dict[str,str]) -> Tuple[Optional[str], Optional[str]]:
        t = c.get("text","").strip()
        ty = c.get("type","fact").lower()
        if not t:
            return (None, None)
        if ty == "classical":
            # 出典照合は青空文庫優先
            return (f'site:aozora.gr.jp "{t}"', f'"{t}" 枕草子 OR 源氏物語 OR 平家物語 OR 徒然草')
        if ty == "statistic":
            return (t + " site:go.jp", t + " 統計 令和")
        if ty == "citation":
            return (f'"{t}" 書籍 出典', f'"{t}" Google Books')
        # 一般 fact（比較主張はunknown判定で止める）
        return (t, None)

    def _dedup_hits(self, arr: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        seen = set(); out=[]
        for h in arr or []:
            u = (h.get("url") or "").split("#")[0]
            if u and u not in seen:
                seen.add(u); out.append(h)
        return out

    def _rule_judge_claim(self, c: Dict[str, Any], hits: List[Dict[str, Any]]) -> Tuple[str, str]:
        txt = c.get("text","")
        ty = c.get("type","fact").lower()

        # 古典は出典存在＝true相当（厳密照合は次版）。誤帰属が見えたらfalse（MVPではunknown止まり）。
        if ty == "classical":
            if any("aozora.gr.jp" in (h.get("url") or "") for h in hits):
                return ("true", "青空文庫に該当あり")
            return ("unknown", "出典未確定（本文照合未了）")

        # 比較・断定一般論は不足スロットで unknown
        if any(k in txt for k in ["より", "優れて", "得意", "最も", "No.1", "世界一"]):
            return ("unknown", "タスク/指標/期間/条件が未指定")

        # 明確な一次情報と矛盾を機械で断定するのは危険 → MVPは conservative
        if not hits:
            return ("unknown", "一次資料が不足")
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

    # ===== 既存の各種ユーティリティ（judge_text, auto_repair など）は原実装のまま =====

    def get_intervention_stats(self) -> Dict[str, Any]:
        try:
            return dict(self._intervention_stats)
        except Exception:
            return {}

    def get_resource_usage(self) -> Dict[str, Any]:
        q = self.rate_limiter.get_current_usage() if hasattr(self, "rate_limiter") else {
            "minute": {"used": 0, "limit": 0, "remaining": 0},
            "daily": {"used": 0, "limit": 0, "remaining": 0},
        }
        return {
            "performance": {"avg_response_time": None, "cache_hit_rate": None, "error_detection_rate": None,},
            "resources": {
                "vram_usage": None,
                "gemini_quota": f"{q['daily']['used']} / {q['daily']['limit']}" if q.get('daily') else None,
                "mcp_calls": "unlimited" if self.use_mcp_search else "disabled",
            },
            "costs": {"current_month": "$0", "projected": "$0"},
        }

    # ===== 追加: 欠落ユーティリティ（最小実装） =====

    def _extract_history_metrics(self, dialogue_context: List[Dict[str, Any]]) -> Optional[Dict[str, int]]:
        """直近の発話から [history-metrics] ブロックや JSON を拾い、簡易メトリクスを返す。
        不在なら簡易ヒューリスティックで classic_refs を概算。
        返却例: {"classic_refs": 3, "kyukokumei_refs": 0, "food_refs": 0}
        """
        try:
            texts: List[str] = []
            for m in dialogue_context[-6:]:  # 直近少数だけ
                if isinstance(m, dict) and m.get("speaker") != "Director":
                    t = m.get("message") or ""
                    if isinstance(t, str) and t:
                        texts.append(t)

            blob = "\n\n".join(texts)
            if not blob:
                return None

            # 1) [history-metrics] ... [/history-metrics]
            m = re.search(r"\[history-metrics\](.+?)\[/history-metrics\]", blob, flags=re.DOTALL | re.IGNORECASE)
            if m:
                raw = m.group(1).strip()
                # JSON 部分だけを抽出
                try:
                    start = raw.find("{")
                    end = raw.rfind("}")
                    if start >= 0 and end > start:
                        raw = raw[start : end + 1]
                    data = json.loads(raw)
                    out = {}
                    for k in ("classic_refs", "kyukokumei_refs", "food_refs"):
                        v = data.get(k)
                        if isinstance(v, int):
                            out[k] = v
                        elif isinstance(v, str) and v.isdigit():
                            out[k] = int(v)
                    return out or None
                except Exception:
                    pass

            # 2) JSON 風の行を直読み
            try:
                m2 = re.search(r"\{[^\}]*classic_refs[^\}]*\}", blob, flags=re.DOTALL)
                if m2:
                    data = json.loads(m2.group(0))
                    out = {}
                    for k in ("classic_refs", "kyukokumei_refs", "food_refs"):
                        v = data.get(k)
                        if isinstance(v, int):
                            out[k] = v
                        elif isinstance(v, str) and v.isdigit():
                            out[k] = int(v)
                    return out or None
            except Exception:
                pass

            # 3) 簡易ヒューリスティック: 古典ワード/引用の出現数
            classics_words = ["枕草子", "源氏物語", "徒然草", "平家物語", "方丈記", "竹取物語"]
            classics_count = 0
            for t in texts:
                classics_count += sum(1 for w in classics_words if w in t)
                classics_count += len(re.findall(r"[「『][^」』]{4,30}[」』]", t))
            return {"classic_refs": classics_count, "kyukokumei_refs": 0, "food_refs": 0}
        except Exception:
            return None

    def _build_pushback_plan(self, target_label: str, pushback_text: str) -> Dict[str, Any]:
        """指摘/確認を促すシンプルな介入プランを生成。"""
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
        """プラン内の max_chars を倍率で調整し、上限でクリップ。"""
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

    def _schedule_soft_ack(self, label: str) -> None:
        """ソフト相槌の予約を登録。既存があれば延長のみ。"""
        try:
            tgt = label if label in ("A", "B") else self._label_of(label)
        except Exception:
            tgt = "A"
        remaining = getattr(self, "soft_ack_max_reminders", 2)
        window = getattr(self, "soft_ack_expire_window", 4)
        expire = int(self.turn_counter) + int(window)
        st = self.pending_soft_ack.get(tgt, {"remaining": 0, "expire_turn": expire})
        st["remaining"] = max(st.get("remaining", 0), int(remaining))
        st["expire_turn"] = max(st.get("expire_turn", expire), expire)
        self.pending_soft_ack[tgt] = st

    def _build_soft_ack_plan(self, target_label: str) -> Dict[str, Any]:
        """短い相槌トーンを示すガイド。"""
        try:
            target = target_label if target_label in ("A", "B") else self._label_of(target_label)
        except Exception:
            target = "A"
        return {
            "turn_style": {
                "speaker": target,
                "length": {"max_chars": 70, "max_sentences": 1},
                "preface": {"aizuchi": True, "aizuchi_list": [random.choice(AIZUCHI_POOL)], "prob": 1.0},
                "speech_act": "agree_short",
                "follow_up": "none",
                "ban": ["praise", "list_format", "long_intro"],
            },
            "cadence": {
                "avoid_consecutive_monologues": True,
                "enforce_question_ratio": list(QUESTION_RATIO_TARGET),
            },
        }

    def _build_refocus_plan(self, theme: str, target_label: str = "B") -> Dict[str, Any]:
        """話題の軌道修正（テーマへ戻す）ための簡潔なガイドを生成。"""
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

# 既存クラス名の互換エイリアス
AutonomousDirector = NaturalConversationDirector
