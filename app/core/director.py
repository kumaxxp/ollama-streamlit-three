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
    # 一般略語・会話頻出語（LLM抽出の誤検出を抑制）
    "AI", "IT", "SNS", "CPU", "GPU", "OS", "PC", "NG", "OK",
    # 会話の汎用語
    "会話", "議題", "話題", "テーマ", "意見", "考え", "気持ち", "全部", "本当", "自然", "数字", "欲求",
    "結局", "古都", "本物", "再現", "データ", "解釈",
])

class RateLimitManager:
    """Gemini API レート制限の簡易管理（オフラインでも動作可能なスタブ）。"""

    def __init__(self):
        # 既定値（無料枠の目安）
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
            # ライブラリ未導入やキー不在時は無効化
            self.enabled = False
            self._model = None

    def add_entries(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """エントリをキューに積み、バッチ到達時のみ処理して結果を返す。"""
        if not entries:
            return []
        self.detection_queue.extend(entries)
        if len(self.detection_queue) < self.batch_size:
            return []
        # 3件まとめて取り出し
        batch = self.detection_queue[: self.batch_size]
        self.detection_queue = self.detection_queue[self.batch_size :]
        return self._process_batch(batch)

    def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """同期処理で簡易実装（UI応答性のため evaluate 側では非ブロッキングに扱う）。"""
        try:
            if not self.enabled or not self._model:
                # 機能無効時は空で返す（介入なし）
                return []
            prompt = (
                "以下の発話を同時に分析し、それぞれについて次を検出しJSONで返してください:\n"
                "1) 事実誤認 2) 論理矛盾 3) ハルシネーション 4) 固有名詞の誤用\n"
                "問題点の検出仕様はそのままに、各発話について発言者に向けた短い『ツッコミ/訂正要求』文も1つ生成してください。\n"
                "出力要件: JSONのみ。pushback は日本語で20-60文字、具体的で端的に（丁寧すぎない助言調、例:『それ名前違う。○○のこと？』）。\n"
                "必ず次の形式で: {\"results\":[{\"id\":str,\"errors\":[{\"type\":str,\"detail\":str,\"entity\":str|null}],\"pushback\":str}]}\n\n"
                + json.dumps(batch, ensure_ascii=False)
            )
            resp = self._model.generate_content(prompt)
            text = getattr(resp, "text", None)
            if isinstance(text, str) and text.strip():
                return self._parse_batch_response(text)
            # fallback: try to stringify response
            try:
                return self._parse_batch_response(str(resp))
            except Exception:
                return []
        except Exception:
            return []

    def _parse_batch_response(self, s: str) -> List[Dict[str, Any]]:
        """Gemini応答テキストから results 配列を抽出して返す。"""
        try:
            # JSON抽出（前後に説明文が混ざる可能性に対応）
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
    """会話のテンポ/尺/相槌を制御するディレクター。内容判断を最小化。"""

    def __init__(
        self,
        ollama_client=None,
        model_name: str = "qwen2.5:7b",
        gemini_api_key: Optional[str] = None,
        use_mcp: bool = True,
        local_model: Optional[str] = None,
    ):
        # LLM(任意)
        self.client = ollama_client
        self.model_name = model_name
        self.temperature = 0.2
        self.current_phase = "flow"  # flow→wrap
        self.turn_counter = 0

        # 参加者名（表示名）-> A/B マッピング補助
        self.participants = []  # [nameA, nameB]

        # メトリクス
        self.metrics = {
            "avg_chars_last3": 0.0,
            "question_ratio": 0.0,
            "consecutive_by_last": 0,
            "last_speaker": None,
            "last_speaker_label": None,
        }

        # v4: 監視/制限/費用最適化
        self.rate_limiter = RateLimitManager()
        self.cost_optimizer = CostOptimizer()
        self.gemini_detector = GeminiErrorDetector(gemini_api_key)

        # v4: MCP/検索
        self.use_mcp_search = bool(use_mcp)
        self.mcp_adapter = MCPWebSearchAdapter(language="ja") if self.use_mcp_search else None

        # v4: エンティティ検証キャッシュ
        self.entity_cache = {}
        # 共有知識キャッシュ（作品・古典など）
        self.shared_knowledge_cache = {}

        # v4: LLMを使ったエンティティ抽出フォールバック（ヒューリスティクス既定OFF/LLM既定ON）
        self.use_llm_entity_fallback = True
        self.use_heuristics = bool(int(os.getenv("DIRECTOR_USE_HEURISTICS", "0")))  # 0=OFF(既定), 1=ON
        self.always_llm_entity_scan = bool(int(os.getenv("DIRECTOR_ALWAYS_LLM_ENTITY_SCAN", "1")))  # 1=ON(既定)
        self.entity_llm_model = local_model or self.model_name

        # v4+: LLM主導の異常検出（ヒューリスティクスを最小化し、AIに「おかしなもの」を網羅抽出させる）
        self.use_llm_anomaly_detector = True
        self.max_anomaly_checks_per_turn = 4

        # v4: Soft Ack 設定
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

        # エンティティ検出の厳格度と1ターン上限
        self.entity_detection_mode = "strict"  # strict | balanced
        self.max_entity_checks_per_turn = 2
        # LLM出力はJSONではなく文章を優先するか（環境変数で切替: 1/0）。既定ON
        try:
            self.prefer_text_llm_output = bool(int(os.getenv("DIRECTOR_PREFER_TEXT_LLM_OUTPUT", "1")))
        except Exception:
            self.prefer_text_llm_output = True

        # リフォーカス（オフトピック）制御
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
        if stats["avg_chars_last3"] > 110:
            return random.choice(["reflect", "agree_short", "handoff"])
        return random.choice(["answer", "reflect", "agree_short", "disagree_short"])

    def _decide_max_chars(self, stats: Dict[str, Any]) -> int:
        # メリハリ: たまにロングターンを許可して深掘りを促す（初期フェーズは上げ目）
        base_prob = 0.22
        if self.turn_counter <= 2:
            base_prob = 0.55
        long_turn = random.random() < base_prob
        if long_turn and stats["question_ratio"] <= QUESTION_RATIO_TARGET[1]:
            # 長め: 120〜160
            return random.choice([130, 140, 150, 160])
        # 直近が長すぎる時は抑制
        if stats["avg_chars_last3"] > 120:
            return 85
        # 短すぎる時は増量
        if stats["avg_chars_last3"] < 45:
            return 120
        # 通常帯
        return 95

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

    # ===== オフトピック検知とリフォーカス =====
    def _detect_offtopic(self, dialogue_context: List[Dict[str, Any]], theme: Optional[str]) -> Tuple[bool, Optional[str]]:
        """簡易ヒューリスティクスでテーマ逸脱を検出。テーマ主要語と直近2発話の主要語の重なりが0ならオフトピック。
        """
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

    def _build_refocus_plan(self, theme: str, target_label: str = "B") -> Dict[str, Any]:
        tip = f"本筋に戻そう。テーマ『{theme}』の中で一つに絞って話そう"
        question = f"じゃあ『{theme}』で、いちばん気になる論点は？"
        plan = {
            "turn_style": {
                "speaker": target_label,
                "length": {"max_chars": 100, "max_sentences": 2},
                "preface": {"aizuchi": True, "aizuchi_list": [tip], "prob": 1.0},
                "speech_act": "ask",
                "follow_up": "none",
                "ban": ["long_intro", "list_format", "praise"],
            },
            "cadence": {"avoid_consecutive_monologues": True},
            "closing_hint": "続ける",
            "note": {"refocus": True, "theme": theme, "reset_prompt": question},
        }
        return plan

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

    # 参加者名の除外用ヘルパー
    def _normalize_name_base(self, name: str) -> str:
        """括弧や敬称を除いたベース名を返す。"""
        if not name:
            return ""
        s = str(name).strip()
        # 括弧注記を除去
        s = s.split("（")[0].split("(")[0].strip()
        # 代表的な敬称・呼び方を末尾から剥がす
        for suf in ["さん", "ちゃん", "くん", "君", "様", "氏", "殿", "せんせい", "先生"]:
            if s.endswith(suf):
                s = s[: -len(suf)]
                break
        return s.strip()

    def _is_participant_name(self, name: str) -> bool:
        """候補名が会話参加者（Agent）の表示名/呼び方と一致するか。"""
        base = self._normalize_name_base(name)
        for p in self.participants or []:
            if self._normalize_name_base(p) == base and base:
                return True
        return False

    # 旧: イベント検出用ヘルパはLLM異常検出に統合したため削除

    # ===== 参考: 既存 evaluate_dialogue の互換API(機能拡張版) =====
    async def evaluate_dialogue(self, dialogue_context: List[Dict[str, Any]], mode: Optional[str] = None) -> Dict[str, Any]:
        """
        旧APIの互換インターフェース。
        優先度:
          1) 固有名詞チェックに基づく訂正促し（entity_correction）
          2) 指摘された側への軽い受け流しのトーン指示（tone_guidance_soft_ack）
          3) リズム/長さ/話法のガイド（length_tempo_speech_act）
        すべて既存のコントローラ/UIに無改変で流せる形式で返す。
        """
        # まず通常のリズム計画を用意（フォールバック用）
        cmd_rhythm = self.plan_next_turn(dialogue_context)
        # 呼び出しカウンタ
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

        # 履歴メトリクス（history-metrics ブロック）を抽出
        metrics = self._extract_history_metrics(dialogue_context)
        if isinstance(metrics, dict):
            debug_info["history_metrics"] = metrics

        # 直近の非Director発話とその一つ前を取得
        last_entry = None
        prev_entry = None
        for m in reversed(dialogue_context):
            if m.get("speaker") and m.get("speaker") != "Director":
                if last_entry is None:
                    last_entry = m
                else:
                    prev_entry = m
                    break

        # 参加者A/Bの特定を更新（_analyzeはplan_next_turn内でも呼ばれているが安全のため）
        _ = self._analyze(dialogue_context)

        # 早期: オフトピック検知 → リフォーカス
        try:
            theme = self.session_theme
            offtopic, reason = self._detect_offtopic(dialogue_context, theme)
        except Exception:
            offtopic, reason = (False, None)
        if offtopic and (self._eval_calls - self._last_refocus_eval >= self._refocus_cooldown):
            target_label = "B"  # Agent2 を既定に
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

        # 0) 文章ベースのホリスティック評価（JSON出力を使わず、短い助言テキストを優先）
        if last_entry and isinstance(last_entry.get("message"), str) and getattr(self, "prefer_text_llm_output", False):
            # 指摘/訂正モードの簡易検出（最後の発話が“確認/訂正/引用控え”を促すタイプか）
            def _seems_correction(text: str) -> bool:
                try:
                    import re as _re
                    t = (text or "").lower()
                    # 和文/英文の代表的な訂正・確認語 + 疑問・確認表現
                    patterns = [
                        r"違|ちが|誤|間違|おかし|本当|ほんとう|根拠|出典|ソース|引用|どこ|ではなく|じゃない|嘘|デマ|検証|確か|wiki|wikipedia|参考|典拠|証拠",
                    ]
                    if _re.search("|".join(patterns), t):
                        return True
                    if "?" in text or "？" in text:
                        # 疑問符単独では弱いが、直前が反論/確認語なら強い
                        return True
                except Exception:
                    pass
                return False

            last_disp = last_entry.get("speaker") or ""
            last_text = last_entry.get("message", "")
            critic_mode = _seems_correction(last_text)
            # オフエンダー = 直前の相手発話（指摘モード時）、通常は最終発話者
            if critic_mode and prev_entry and prev_entry.get("speaker"):
                offender_disp = prev_entry.get("speaker")
                offender_label = self._label_of(offender_disp)
                target_label = self._label_of(last_disp)  # 指摘側に“穏やかな確認/一言質問”を促す
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
                    # もしレビューが検索キーワードを含むなら、その場でMCPで軽い検索を実施し共有用に格納
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
                            # 追加: 地理座標が取れれば日本内外の簡易判定を付与
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
                    # 追加: 作品名の簡易検出（引用過多の抑制や要旨確認の誘導）
                    try:
                        works = self._detect_classics_and_works(text0) or []
                        # 追加: ホリスティックレビュー本文内の「…」/『…』も検索対象へ
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
                    # 作品タイトル（複数）と、レビュー内の任意クエリでWikipediaスニペット検索
                    try:
                        wiki_snippets = []
                        if self.use_mcp_search and self.mcp_adapter:
                            # まず作品タイトルそれぞれで検索
                            for w in works[:3]:
                                snips = self.mcp_adapter.search_snippets(w, limit=2) or []
                                for s in snips:
                                    s_row = {"query": w, "title": s.get("title"), "url": s.get("url"), "excerpt": s.get("excerpt")}
                                    wiki_snippets.append(s_row)
                            # 追加: レビューが示唆する queries でも1本だけ検索
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
                    # Agent向け短期ディレクティブを生成（2ターン有効）
                    review_directives = {
                        "required_actions": [],
                        "ensure": ["one_concise_question"],
                        "avoid": ["praise", "list_format"],
                        "tone_hint": "端的/助言調",
                        "ttl_turns": 2,
                    }
                    # 強い引用禁止（履歴で古典引用が多すぎる場合）
                    try:
                        cref = int((metrics or {}).get("classic_refs", 0))
                        if cref >= 12:
                            review_directives["required_actions"].append("ban_classic_quotes")
                            if "overuse_classics" not in review_directives["avoid"]:
                                review_directives["avoid"].append("overuse_classics")
                    except Exception:
                        pass
                    # 検索情報があればURLを1つだけ引用可
                    if isinstance(debug_info.get("research"), list) and debug_info["research"]:
                        review_directives["required_actions"].append("cite_one_url_if_available")
                    # 日本外の可能性があれば地理ツッコミを促す
                    geo = debug_info.get("geo") or {}
                    if isinstance(geo, dict) and any((isinstance(g, dict) and g.get("in_japan") is False) for g in geo.values()):
                        review_directives["required_actions"].append("geo_nudge")
                    # 作品名があれば要旨確認と引用控えめ
                    if works:
                        review_directives["required_actions"].append("summarize_work")
                        review_directives["avoid"].append("overuse_classics")
                        debug_info["works_detected"] = []
                        # 各作品を即時に検証し、存在しない/曖昧は NG/AMBIGUOUS として注記
                        verified_found = False
                        for w in works[:3]:
                            try:
                                row = self._summarize_or_validate_work(w)
                            except Exception:
                                row = {"title": w, "verdict": "AMBIGUOUS", "url": None, "summary": None}
                            debug_info["works_detected"].append(row)
                            if row.get("verdict") in ("NG", "AMBIGUOUS"):
                                # NG/曖昧なら、誤記/架空の可能性を短く確認する行動を促す
                                if "offer_correction" not in review_directives["required_actions"]:
                                    review_directives["required_actions"].append("offer_correction")
                            elif row.get("verdict") == "VERIFIED":
                                verified_found = True
                        # VERIFIED が含まれる場合は、文脈との関連性を確認し、無関係なら強めの指摘を許可
                        if verified_found:
                            if "check_verified_relevance" not in review_directives["required_actions"]:
                                review_directives["required_actions"].append("check_verified_relevance")
                            # 口調ヒントを少し強めに
                            review_directives["tone_hint"] = review_directives.get("tone_hint") or "端的/少し強めに"
                    # 1回目のプラン
                    plan = self._build_holistic_intervention_text(target_label, htx)
                    # 指摘モードなら、話法を穏やかな確認（ask）寄りに補正
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
                    # 証拠(wiki_snippets)があるなら、2回目のホリスティック再レビューで指示を強化
                    try:
                        wiki_snips = debug_info.get("wiki_snippets") or []
                        if wiki_snips:
                            second = self._llm_holistic_review_with_evidence(text0, wiki_snips[:3])
                            if isinstance(second, dict):
                                # pushback の上書き(あれば)
                                pb = (second.get("pushback") or "").strip()
                                if pb:
                                    # planの相槌を差し替え
                                    try:
                                        plan_ts = plan.get("turn_style", {})
                                        pre = plan_ts.get("preface", {})
                                        pre["aizuchi_list"] = [self.auto_repair(pb, max_chars=70)]
                                        pre["aizuchi"] = True
                                        plan_ts["preface"] = pre
                                        plan["turn_style"] = plan_ts
                                    except Exception:
                                        pass
                                # agent_directives の取り込み（元の review_directives とマージ）
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
                    # ツッコミ材料（research/wiki_snippets/works）あり→長さを約3倍に拡張（上限260）
                    try:
                        has_materials = bool(debug_info.get("research") or debug_info.get("wiki_snippets") or debug_info.get("works_detected"))
                        if has_materials:
                            plan = self._scale_plan_length(plan, factor=3, cap=260)
                    except Exception:
                        pass
                    if self.soft_ack_enabled:
                        # オフエンダー側には軽い受け流し（soft-ack）を予定
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
            # リスクがなければテンポ制御のみで返す（重い検出はスキップ）
            # ただし、かっこ内の引用（作品/言い回し）がある場合は最低限の検証を実施し、NG/曖昧なら短くツッコむ
            try:
                works = self._detect_classics_and_works(text0) or []
                # 追加: ホリスティックレビュー本文内の「…」/『…』も検索対象へ
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
                    # まずスニペット検索（ログ用）
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
                    # 判定
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
                        # 指摘モードなら、次話者=指摘側。それ以外は通常どおり交代
                        if critic_mode and last_entry and last_entry.get("speaker"):
                            target_label = self._label_of(last_entry.get("speaker"))
                        else:
                            target_label = self._other(self._label_of(offender_disp))
                        # 短いツッコミを促すディレクティブ
                        review_directives = {
                            "required_actions": ["offer_correction", "one_concise_question"],
                            "avoid": ["praise", "list_format", "overuse_classics"],
                            "tone_hint": "端的/助言調",
                            "ttl_turns": 2,
                        }
                        pb = "そんな言葉ないですよ。別の表現のこと？"
                        plan = self._build_pushback_plan(target_label, pb)
                        # 指摘モード時は、より穏やかな確認トーンに補正
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
                        # 引用誤りのツッコミ材料あり→長さを拡張
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
            # 全て VERIFIED で安全だが、引用が文脈に無関係な可能性がある→テンポ制御 + 関連性チェック指示を返す
                self._intervention_stats["rhythm_guides"] += 1
                # レビュー指示: verified の関連性チェック（あれば） + 強い引用禁止（必要時）
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

    # 1) 固有名詞チェック → 訂正促し（指摘するのは「発言者と別のAI」= 直近発話者の反対側）
        if last_entry and isinstance(last_entry.get("message"), str):
            offender_disp = last_entry.get("speaker") or ""
            offender_label = self._label_of(offender_disp)
            target_label = self._other(offender_label)
            text = last_entry.get("message", "")

            # 1) ヒューリスティクス抽出（無効化設定ならスキップ）
            entities: List[Dict[str, Any]] = []
            if getattr(self, "use_heuristics", False):
                entities = self._extract_entities(text)
                if entities:
                    logger.debug(f"Director entity candidates: {entities}")
                    debug_info["heuristic_entities"] = entities

            # 2) LLM 抽出を併用（人物に限らず、未知用語も拾う）
            llm_entities: List[Dict[str, Any]] = []
            if self.use_llm_entity_fallback and self.client is not None and (
                self.always_llm_entity_scan or not entities
            ):
                scan = self._llm_scan_entities(text)
                if isinstance(scan, dict):
                    llm_entities = [
                        {"name": (e.get("name") or "").strip(), "type": (e.get("type") or "OTHER").upper()}
                        for e in (scan.get("entities") or [])
                        if isinstance(e, dict) and (e.get("name") or "").strip()
                    ]
                    if llm_entities:
                        logger.debug(f"LLM entity candidates: {llm_entities}")
                        debug_info["llm_entities"] = llm_entities

            # 3) 候補集合を決定（LLM優先。ヒューリスティクスは無効時は使わない）
            merged: List[Dict[str, Any]] = []
            seen = set()
            for src in (llm_entities if not getattr(self, "use_heuristics", False) else (entities + llm_entities)):
                n = src.get("name")
                if n and n not in seen:
                    seen.add(n)
                    merged.append(src)

            # 4) 候補選定（複数対応）: LLM抽出を優先（PERSON優先）。一般語は強く除外
            selected: List[Dict[str, Any]] = []
            # LLM異常検出（任意）
            anomalies: List[Dict[str, Any]] = []
            # 共通フィルタ
            def _passes(name: str, etype: str) -> bool:
                if not name:
                    return False
                n = name.strip()
                if self._is_participant_name(n):
                    return False
                if n in STOP_ENTITY_COMMON:
                    return False
                import re as _re
                if _re.fullmatch(r"[A-Z]{1,3}", n):
                    return False
                if _re.search(r"(ちゃん|たん|っち|にゃん)$", n):
                    return False
                fiction_keywords = ["キャラ", "キャラクター", "VTuber", "ゆるキャラ", "マスコット", "アバター", "二次元", "三次元化", "推し", "擬人化"]
                if any(kw in n for kw in fiction_keywords):
                    return False
                # 厳格モード: PERSON 以外は明確な固有名手掛かりが無いと弾く
                if (etype or "").upper() == "PERSON":
                    # 英文名 or 漢字2文字以上の氏名想定
                    if " " in n:
                        return True
                    if _re.search(r"[\u4E00-\u9FFF]", n) and len(n) >= 2:
                        return True
                    return False
                # 非PERSONは以下の強いシグナルのいずれかが必要
                # 1) 著名性ヒントが直近発話にある
                if self._has_prominence_hint(text):
                    return True
                # 2) 組織/地名などの典型サフィックス
                org_suffix = ["株式会社", "大学", "省", "庁", "市", "区", "県", "都", "府", "政府", "内閣", "党", "新聞", "テレビ", "放送", "病院", "研究所", "学会", "博物館", "神社", "寺", "駅"]
                if any(n.endswith(suf) for suf in org_suffix):
                    return True
                # 3) 長めのカタカナ語（一般語を避けるため5文字以上）
                if _re.fullmatch(r"[\u30A1-\u30F6\u30FC]{5,}", n):
                    return True
                # 4) 英数混在（製品名など）
                if _re.search(r"[A-Za-z].*[0-9]|[0-9].*[A-Za-z]", n):
                    return True
                return False

            # 4-a) AI主導の異常検出で「おかしなもの」を網羅抽出
            pending_claims: List[Dict[str, Any]] = []
            if self.use_llm_anomaly_detector and self.client is not None:
                try:
                    anomalies = self._llm_detect_anomalies(text) or []
                    if anomalies:
                        debug_info["anomalies"] = anomalies
                        # 検証候補（PERSON/ORG/EVENT/ENTITY系）と、即時指摘（CLAIM系）に仕分け
                        for a in anomalies:
                            label = (a.get("label") or a.get("name") or "").strip()
                            kind = str(a.get("kind") or a.get("type") or "OTHER").upper()
                            reason = a.get("reason") or a.get("why") or None
                            if not label:
                                continue
                            if kind == "CLAIM":
                                pending_claims.append({
                                    "name": label,
                                    "type": "CLAIM",
                                    "verdict": "AMBIGUOUS",
                                    "evidence": None,
                                    "evidence_text": reason,
                                })
                                continue
                            mapped = "ENTITY"
                            if kind in ("PERSON", "ORG", "EVENT"):
                                mapped = kind
                            # TERM/OTHERもENTITYとして扱う
                            if all(c.get("name") != label for c in selected) and _passes(label, mapped):
                                selected.append({"name": label, "type": mapped})
                except Exception:
                    pass

            ordered_sources: List[Dict[str, Any]] = []
            if llm_entities:
                ordered_sources.extend([e for e in llm_entities if (e.get("type") or "").upper() == "PERSON"])
                ordered_sources.extend([e for e in llm_entities if (e.get("type") or "").upper() != "PERSON"])
            elif merged:
                ordered_sources = merged
            for e in ordered_sources:
                n = (e.get("name") or "").strip()
                et = (e.get("type") or "ENTITY").upper()
                if _passes(n, et):
                    selected.append({"name": n, "type": et})
            if getattr(self, "use_heuristics", False) and merged:
                for e in merged:
                    n = (e.get("name") or "").strip()
                    et = (e.get("type") or "ENTITY").upper()
                    if _passes(n, et) and all(c["name"] != n for c in selected):
                        selected.append({"name": n, "type": et})

            # 1ターンの検証数を制限
            if selected:
                cap = max(self.max_entity_checks_per_turn, self.max_anomaly_checks_per_turn)
                selected = selected[: cap]

            if selected:
                debug_info["all_candidates"] = list(selected)
                verifications: List[Dict[str, Any]] = []
                problem_names: List[str] = []
                # 先にCLAIM系の擬似検証を反映（LLM理由を evidence_text に格納）
                for pv in pending_claims:
                    verifications.append(dict(pv))
                    problem_names.append(pv.get("name"))
                for c in selected:
                    name = c.get("name")
                    etype = c.get("type", "ENTITY")
                    logger.info(f"MCP verify check target='{name}' type={etype}")
                    verdict = self._verify_entity(name, etype)
                    evidence = None
                    evidence_text = None
                    ec = self.entity_cache.get(name)
                    if isinstance(ec, dict):
                        evidence = ec.get("evidence")
                        evidence_text = ec.get("evidence_text")
                    # 追加: 地理候補であれば座標も試行（雑にENTITY全般で試し、日本内外を持つ）
                    try:
                        if self.use_mcp_search and self.mcp_adapter and etype in ("EVENT", "ENTITY"):
                            lat, lon, in_jp = self.mcp_adapter.get_coordinates(name)
                            if lat is not None and lon is not None:
                                debug_info.setdefault("geo", {})[name] = {
                                    "lat": lat, "lon": lon, "in_japan": in_jp
                                }
                    except Exception:
                        pass
                    vrow = {"name": name, "type": etype, "verdict": verdict, "evidence": evidence, "evidence_text": evidence_text}
                    verifications.append(vrow)
                    if verdict in ("AMBIGUOUS", "NG"):
                        problem_names.append(name)
                # 互換性のため、単一表示用の primary を設定
                debug_info["verifications"] = verifications
                if verifications:
                    # 問題があればそれを、なければ先頭を primary に
                    primary = next((v for v in verifications if v.get("verdict") in ("AMBIGUOUS", "NG")), verifications[0])
                    debug_info["selected_candidate"] = {"name": primary.get("name"), "type": primary.get("type")}
                    debug_info["verification"] = {
                        "verdict": primary.get("verdict"),
                        "evidence": primary.get("evidence"),
                        "evidence_text": primary.get("evidence_text"),
                    }

                if problem_names:
                    self._intervention_stats["entity_checks"] += len(selected)
                    plan = self._build_entity_correction_plan(target_label, problem_names)
                    # エンティティ誤用の具体材料あり→長さを拡張
                    plan = self._scale_plan_length(plan, factor=3, cap=260)
                    if self.soft_ack_enabled:
                        self._schedule_soft_ack(offender_label)
                        self._intervention_stats["soft_ack_dispatched"] += 1
                    self._intervention_stats["entity_corrections"] += 1
                    return {
                        "intervention_needed": True,
                        "reason": "entity_check",
                        "intervention_type": "entity_correction",
                        "message": json.dumps(plan, ensure_ascii=False),
                        "response_length_guide": "簡潔",
                        "confidence": 0.75,
                        "director_debug": debug_info,
                    }

                # v4: バランス/徹底モードではGeminiバッチに投入（APIキーがあれば）
                try:
                    if self.gemini_detector and self.gemini_detector.enabled and mode in ("balanced", "thorough"):
                        can_cloud, _ = self.rate_limiter.should_use_cloud()
                        if can_cloud:
                            batch_size = self.cost_optimizer.get_optimal_batch_size(None)
                            self.gemini_detector.batch_size = max(1, batch_size)
                            results = self.gemini_detector.add_entries([
                                {
                                    "id": f"turn-{self.turn_counter}",
                                    "speaker": offender_disp,
                                    "text": text,
                                }
                            ])
                            if results:
                                self._intervention_stats["gemini_batches"] += 1
                                self.rate_limiter.note_call(1)
                                debug_info["gemini"] = results
                                # 直近エントリの結果を参照
                                for r in results:
                                    if r.get("id") == f"turn-{self.turn_counter}":
                                        errs = r.get("errors") or []
                                        pushback_txt = (r.get("pushback") or "").strip()
                                        # 固有名詞誤用などがあれば軽い指摘を優先
                                        err_entity = None
                                        for e in errs:
                                            if (e.get("type") or "").lower() in ("entity", "named_entity", "proper_noun"):
                                                err_entity = e.get("entity")
                                                break
                                        if errs:
                                            # pushback があればそれを優先してツッコミ/訂正要求の発話を誘導
                                            if pushback_txt:
                                                plan = self._build_pushback_plan(target_label, pushback_txt)
                                            else:
                                                entity_hint = err_entity if err_entity else "該当の名称"
                                                plan = self._build_entity_correction_plan(target_label, entity_hint)
                                            # Gemini検出の材料あり→長さを拡張
                                            plan = self._scale_plan_length(plan, factor=3, cap=260)
                                            if self.soft_ack_enabled:
                                                self._schedule_soft_ack(offender_label)
                                                self._intervention_stats["soft_ack_dispatched"] += 1
                                            self._intervention_stats["entity_corrections"] += 1
                                            return {
                                                "intervention_needed": True,
                                                "reason": "gemini_error_detected",
                                                "intervention_type": "entity_correction",
                                                "message": json.dumps(plan, ensure_ascii=False),
                                                "response_length_guide": "簡潔",
                                                "confidence": 0.7,
                                                "director_debug": debug_info,
                                            }
                except Exception:
                    pass

        # 2) Soft Ack（軽い受け流し）配布（ターゲット= offender 本人）
        if self.soft_ack_enabled and self.pending_soft_ack:
            # 期限内かつ残数>0のものを1件配布
            now_turn = self.turn_counter
            for label, st in list(self.pending_soft_ack.items()):
                remaining = st.get("remaining", 0)
                expire = st.get("expire_turn", 0)
                if remaining > 0 and now_turn <= expire:
                    plan = self._build_soft_ack_plan(label)
                    # 消費
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
                # 期限切れは掃除
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

    # ===== v4: 追加の公開ユーティリティ =====
    def generate_length_instruction(self, guide_key: str) -> str:
        """config/director_prompts.json の response_length_guidelines から指示文を取得。"""
        try:
            path_candidates = [
                os.path.join(os.getcwd(), "config", "director_prompts.json"),
                os.path.join(os.path.dirname(__file__), "..", "..", "config", "director_prompts.json"),
            ]
            cfg = None
            for p in path_candidates:
                try:
                    if os.path.exists(p):
                        with open(p, "r", encoding="utf-8") as f:
                            cfg = json.load(f)
                            break
                except Exception:
                    continue
            if not isinstance(cfg, dict):
                return "次は100-160文字、自然な会話文で簡潔に。"
            g = (cfg.get("response_length_guidelines") or {}).get(guide_key) or {}
            instr = g.get("instruction") or "100-160文字で、自然な会話文で。"
            return str(instr)
        except Exception:
            return "100-160文字で、自然な会話文で。"

    def should_end_dialogue(self, dialogue_history: List[Dict[str, Any]], turn_count: int) -> Tuple[bool, str]:
        """終了判定（軽量ヒューリスティクス）。"""
        # ターンが十分 & 質問比率が妥当なら終了提案
        stats = self._analyze(dialogue_history)
        if turn_count >= 18 and QUESTION_RATIO_TARGET[0] <= stats.get("question_ratio", 0.0) <= QUESTION_RATIO_TARGET[1]:
            return True, "十分なターンと質疑の往復が成立したため"
        if turn_count >= 22:
            return True, "上限ターン到達"
        return False, "継続"

    def get_intervention_stats(self) -> Dict[str, Any]:
        try:
            return dict(self._intervention_stats)
        except Exception:
            return {}

    def get_resource_usage(self) -> Dict[str, Any]:
        """監視用の使用状況ダイジェスト（軽量）。"""
        q = self.rate_limiter.get_current_usage() if hasattr(self, "rate_limiter") else {
            "minute": {"used": 0, "limit": 0, "remaining": 0},
            "daily": {"used": 0, "limit": 0, "remaining": 0},
        }
        return {
            "performance": {
                "avg_response_time": None,
                "cache_hit_rate": None,
                "error_detection_rate": None,
            },
            "resources": {
                "vram_usage": None,  # 取得しない（外部依存回避）
                "gemini_quota": f"{q['daily']['used']} / {q['daily']['limit']}" if q.get('daily') else None,
                "mcp_calls": "unlimited" if self.use_mcp_search else "disabled",
            },
            "costs": {
                "current_month": "$0",
                "projected": "$0",
            },
        }

    # ===== 監視LLM: 履歴メトリクス計測 =====
    def build_history_metrics_block(self, last5_texts: List[str]) -> Optional[str]:
        """直近5ターンのテキストを監視LLMに渡し、指定フォーマットのメトリクスブロックを作らせる。
        出力例:
        [history-metrics]\nclassic_refs=2\nkyukokumei_refs=3\nfood_refs=1\n[/history-metrics]
        """
        try:
            if not self.client:
                return None
            joined = "\n".join([str(t) for t in last5_texts if str(t).strip()])
            system = (
                "あなたは会話の監視LLMです。これから与える直近5ターンの会話テキストを読み、"
                "次の3つの出現回数（整数）を数えて、指定フォーマットのみで出力してください。\n"
                "- classic_refs: 古典・典籍・古典文学・古文の引用（例: 源氏物語、枕草子、徒然草、論語、平家物語 など）\n"
                "- kyukokumei_refs: 旧国名の出現（例: 美濃、尾張、越前、近江、甲斐、武蔵 など。『〜国』の『国』は出力禁止）\n"
                "- food_refs: 食文化・料理・味覚を用いたたとえ（例: 出汁、旨味、スパイシー、寿司、ラーメン、カレー などの比喩）\n"
                "厳守: 下の形式のみを出力し、前後に余計な説明を一切付けない。数値は必ず整数。"
            )
            user = (
                "会話（直近5ターン以内・テキスト）:\n" + joined + "\n\n"
                "出力フォーマット:\n"
                "[history-metrics]\n"
                "classic_refs=〈整数〉\n"
                "kyukokumei_refs=〈整数〉\n"
                "food_refs=〈整数〉\n"
                "[/history-metrics]"
            )
            resp = self.client.chat(
                model=self.entity_llm_model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                options={"temperature": 0, "num_ctx": 4096, "num_batch": 128},
                stream=False,
            )
            content = (resp.get("message") or {}).get("content")
            if not content:
                return None
            block = str(content).strip()
            # 最低限のバリデーション（3行の整数）
            import re as _re
            m = _re.search(r"\[history-metrics\][\s\S]*?\[/history-metrics\]", block)
            if not m:
                return None
            blk = m.group(0)
            if not _re.search(r"classic_refs=\d+", blk):
                return None
            if not _re.search(r"kyukokumei_refs=\d+", blk):
                return None
            if not _re.search(r"food_refs=\d+", blk):
                return None
            return blk
        except Exception:
            return None

    def degrade_gracefully(self, remaining_quota: int) -> Dict[str, Any]:
        """残量に応じた機能制限ポリシー（参考実装）。"""
        if remaining_quota > 500:
            return {"all_features": True}
        elif remaining_quota > 100:
            return {
                "entity_check": True,
                "fact_check": False,
                "contradiction": True,
            }
        else:
            return {
                "entity_check": True,
                "fact_check": False,
                "contradiction": False,
            }

    # ===== 固有名詞チェック/Soft Ack 補助 =====
    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """最小ヒューリスティクスで固有名詞候補を抽出。
        - ラテン系氏名: "John Smith" のような先頭大文字2語以上
        - 日本語氏名（漢字連続）: 2-6 文字 + 後続助詞/敬称（が/は/の/さん/氏 等）
        - カタカナ連続: 3文字以上（精度低いので低優先）
        返り値: [{name, type}]
        """
        results: List[Dict[str, str]] = []
        try:
            # 先にイベント句（例: 本能寺の変 等）を優先抽出
            for m in re.finditer(r"([\u4E00-\u9FFF]{2,10}の(?:変|乱|戦い|戦争|合戦)|[\u4E00-\u9FFF]{2,10}(?:事件|騒動|大戦|虐殺|蜂起|暴動|革命))", text):
                cand = m.group(0).strip()
                if cand:
                    results.append({"name": cand, "type": "EVENT"})
            # 英文氏名っぽい
            for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text):
                cand = m.group(1).strip()
                if cand:
                    results.append({"name": cand, "type": "PERSON"})
            # 日本語: 漢字（2-6文字）+ 助詞/敬称の直前
            # PERSON: 敬称が付く場合のみ
            kanji_person_pat = r"([\u4E00-\u9FFF]{2,6})(?=(?:さん|氏|公|卿|殿|様)[がはのをにとへ、。\s])"
            for m in re.finditer(kanji_person_pat, text):
                cand = m.group(1).strip()
                if cand:
                    results.append({"name": cand, "type": "PERSON"})
            # ENTITY: 敬称なし（一般名詞の誤検出を回避するためストップワード除外）
            kanji_entity_pat = r"([\u4E00-\u9FFF]{2,6})(?=[がはのをにとへ、。\s])"
            for m in re.finditer(kanji_entity_pat, text):
                cand = m.group(1).strip()
                if cand and cand not in STOP_ENTITY_COMMON:
                    results.append({"name": cand, "type": "ENTITY"})
            # 未知語補助: 終端がかな等でも拾う（例: 「比重追跡砲みたい」など）
            kanji_unknown_pat = r"([\u4E00-\u9FFF]{3,8})(?=(?:[ぁ-んァ-ン]|$))"
            for m in re.finditer(kanji_unknown_pat, text):
                cand = m.group(1).strip()
                if cand and cand not in STOP_ENTITY_COMMON:
                    results.append({"name": cand, "type": "ENTITY"})
            # カタカナ語（低優先）
            for m in re.finditer(r"[\u30A1-\u30F6\u30FC]{3,}", text):
                cand = m.group(0)
                # 明らかな一般語は除外したいが最小実装ではそのまま
                if cand and len(cand) >= 3:
                    # 種別は未判定
                    results.append({"name": cand, "type": "ENTITY"})
        except Exception:
            pass
        # 重複除去（先頭優先）
        seen = set()
        unique: List[Dict[str, str]] = []
        for r in results:
            key = r.get("name")
            if key and key not in seen:
                seen.add(key)
                unique.append(r)
        return unique[:5]

    def _has_prominence_hint(self, text: str) -> bool:
        """著名性や役職を示すヒント語が含まれるかの簡易判定。"""
        hints = [
            "有名", "著名", "俳優", "女優", "歌手", "大統領", "首相", "CEO", "創業者", "ノーベル", "受賞",
        ]
        return any(h in text for h in hints)

    def _verify_entity(self, name: str, etype: str) -> str:
        """エンティティの妥当性を判定する最小実装。
        - use_mcp_search が有効ならMCP/検索アダプタでWeb検証を行う
        - 既定は保守的に "AMBIGUOUS"（即断の誤検知を避ける）
        戻り値: 'VERIFIED' | 'AMBIGUOUS' | 'NG'
        """
        cache = self.entity_cache.get(name)
        if cache:
            return cache.get("verdict", "AMBIGUOUS")

        verdict = "AMBIGUOUS"
        evidence = None
        evidence_text = None

        if self.use_mcp_search and self.mcp_adapter:
            try:
                # 既存キャッシュにテキストがなければ詳細版を取得
                v, url, extract = self.mcp_adapter.verify_entity_detail(name, etype)
                # アダプタの結果を標準化
                if v in ("VERIFIED", "AMBIGUOUS", "NG"):
                    verdict = v
                    evidence = url
                    evidence_text = extract
                logger.info(
                    "MCP result name='%s' verdict=%s evidence=%s",
                    name,
                    verdict,
                    evidence,
                )
            except Exception:
                logger.warning("MCP search failed for '%s'", name, exc_info=True)
        else:
            logger.debug("MCP disabled or adapter unavailable; fallback to AMBIGUOUS")

        self.entity_cache[name] = {
            "verdict": verdict,
            "ts": datetime.now().isoformat(),
            "evidence": evidence,
            "evidence_text": evidence_text,
        }
        return verdict

    def _llm_scan_entities(self, text: str) -> Optional[Dict[str, Any]]:
        """LLMに発話から固有名詞（特にPERSON）抽出を依頼し、JSONで受け取るフォールバック。
        期待出力:
        { "entities": [ {"name": str, "type": "PERSON|ORG|OTHER"}, ... ] }
        """
        if not self.client:
            return None
        try:
            # 参加者名（敬称含む）を抽出対象から除外する注記を付加
            participants = [p for p in (self.participants or []) if p]
            part_note = "、".join(participants) if participants else "(会話参加者名は未特定)"
            system = (
                "あなたは対話監督の補助です。以下の日本語発話から 固有名詞 や 不明語(造語/誤用の可能性がある名詞句) を抽出します。"
                "ただし次は抽出しないでください: (1) 一般語・略語（例: AI, IT, SNS, OK, NG など）、(2) テーマ/話題/意見などの汎用語、"
                f"(3) 会話参加者（{part_note}）の名前やそれらへの呼びかけ（〜さん/〜ちゃん/〜くん 等）。"
                "抽出対象は、現実世界の実在が想定される 公的人物(PERSON) や 組織/製品/地名/イベント等(ORG/OTHER) に限ります。"
                "人名(PERSON)を優先しつつ、曖昧だが実在可能性のある専門用語は OTHER にしてください。 出力は次のJSONのみで、説明文は一切不要です。\n"
                "{\n  \"entities\": [ {\"name\": string, \"type\": \"PERSON|ORG|OTHER\"} ... ]\n}"
            )
            user = f"発話:\n{text}\n\nJSONのみを出力してください。"
            resp = self.client.chat(
                model=self.entity_llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                options={"temperature": 0.1, "num_ctx": 4096, "num_batch": 128},
                stream=False,
            )
            content = (resp.get("message") or {}).get("content")
            if not content:
                return None
            # JSON抽出（前後にノイズがあれば抜き出し）
            content_str = str(content).strip()
            start = content_str.find("{")
            end = content_str.rfind("}")
            if start >= 0 and end > start:
                content_str = content_str[start:end+1]
            data = json.loads(content_str)
            if isinstance(data, dict) and "entities" in data:
                return data
        except Exception:
            return None
        return None

    def _llm_holistic_review_with_evidence(self, text: str, evidence: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Wikipedia等のスニペット(タイトル/URL/要約)を添えて、再度ホリスティックレビューを実行。
        期待出力(JSON):
        {
          "issues": [{"type":"fact|logic|entity|geo|style", "message": str}],
          "pushback": str,
          "agent_directives": {
             "required_actions": [str...],
             "ensure": [str...],
             "avoid": [str...],
             "tone_hint": str
          }
        }
        """
        if not self.client or not evidence:
            return None
        try:
            # 証拠を短く整形
            lines = []
            for i, ev in enumerate(evidence[:3], 1):
                t = (ev.get("title") or "").strip()
                u = (ev.get("url") or "").strip()
                ex = (ev.get("excerpt") or "").strip()
                if t or u or ex:
                    ex_short = ex.replace("\n", " ")[:240]
                    lines.append(f"[{i}] {t} | {u} | {ex_short}")
            ev_text = "\n".join(lines)

            participants = [p for p in (self.participants or []) if p]
            part_note = "、".join(participants) if participants else None
            system = (
                "あなたは対話監督のレビュアです。以下の発話と参考情報(要約/URL)を踏まえて、\n"
                "おかしさの指摘・短いツッコミ(pushback)・次ターン用の行動指示(agent_directives)をJSONで返してください。\n"
                "行動指示は『1つだけ短い質問』『可能ならURLを1つだけ』『誤記疑いならやわらかく確認』『引用控えめ』などを含めてください。"
            )
            if part_note:
                system += (
                    f"\n参加者: {part_note}\n"
                    "注意: 上記参加者名やそれらの呼称/あだ名/短縮形は会話上の相手を指すものとして扱い、"
                    "『誰を指すか不明』の指摘対象に含めない。"
                )
            user = (
                "発話:\n" + text + "\n\n"
                "参考情報(上位):\n" + ev_text + "\n\n"
                "JSONのみで出力:\n"
                "{\n  \"issues\": [{\"type\": \"fact|logic|entity|geo|style\", \"message\": string}],\n"
                "  \"pushback\": string,\n  \"agent_directives\": {\n"
                "    \"required_actions\": [string], \"ensure\": [string], \"avoid\": [string], \"tone_hint\": string\n  }\n}"
            )
            resp = self.client.chat(
                model=self.entity_llm_model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                options={"temperature": 0.1, "num_ctx": 4096, "num_batch": 128},
                stream=False,
            )
            content = (resp.get("message") or {}).get("content")
            if not content:
                return None
            s = str(content).strip()
            a = s.find("{"); b = s.rfind("}")
            if a >= 0 and b > a:
                s = s[a:b+1]
            data = json.loads(s)
            if isinstance(data, dict):
                return data
        except Exception:
            return None
        return None

    def _detect_classics_and_works(self, text: str) -> List[str]:
        """古典・作品名っぽい語を簡易検出（例: 源氏物語/枕草子/徒然草/論語/平家物語 など）。"""
        try:
            cand: List[str] = []
            # 代表的パターン: 〜物語/〜草子/〜集/〜記/論語 ほか
            pats = [
                r"[\u4E00-\u9FFF]{1,6}物語",
                r"[\u4E00-\u9FFF]{1,6}草子",
                r"[\u4E00-\u9FFF]{1,6}集",
                r"[\u4E00-\u9FFF]{1,6}記",
                r"論語",
            ]
            import re as _re
            for p in pats:
                for m in _re.finditer(p, text):
                    w = m.group(0)
                    if w and w not in cand:
                        cand.append(w)
            # かっこ内の引用タイトルを追加検出（「…」/『…』）
            try:
                for m in _re.finditer(r"[「『]([^」』]{2,20})[」』]", text):
                    t = (m.group(1) or "").strip()
                    # 記号だらけ/短すぎ/長すぎを除外
                    if t and 2 <= len(t) <= 20:
                        cand.append(t)
            except Exception:
                pass
            # 既知語の典型例を優先
            known = ["源氏物語", "枕草子", "徒然草", "平家物語", "論語"]
            for k in known:
                if k in text and k not in cand:
                    cand.append(k)
            return cand[:5]
        except Exception:
            return []

    def _summarize_or_validate_work(self, title: str) -> Dict[str, Any]:
        """作品タイトルについてMCPで要約/存在判定し、キャッシュして返す。"""
        if not self.use_mcp_search or not self.mcp_adapter:
            return {"title": title, "verdict": "AMBIGUOUS"}
        if title in self.shared_knowledge_cache:
            return self.shared_knowledge_cache[title]
        v, u, ex = self.mcp_adapter.verify_entity_detail(title, "ENTITY")
        row = {"title": title, "verdict": v, "url": u, "summary": ex}
        self.shared_knowledge_cache[title] = row
        return row

    def _llm_detect_anomalies(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """LLMに『おかしなもの（誤用/曖昧/疑わしい主張/架空名等）』の網羅抽出を依頼し、
        次のJSON形式で返す: {"anomalies":[{"label":str,"kind":"PERSON|ORG|EVENT|ENTITY|CLAIM|OTHER","reason":str}...]}
        最小限の実装として、Ollama互換のchat APIにプロンプトしてJSON抽出する。
        """
        if not self.client:
            return None
        try:
            system = (
                "あなたは会話監督の補助です。以下の日本語発話から、内容面で『おかしなもの』を網羅抽出してください。"
                "対象例: 架空/誤用の可能性がある固有名詞、曖昧な主張(年代・場所・出来事が不明確)、不自然な専門用語の組み合わせなど。"
                "出力はJSONのみ。各項目は label(短い名称), kind(PERSON/ORG/EVENT/ENTITY/CLAIM/OTHER), reason(一言理由) を含めてください。"
                "説明文や前置きは不要です。"
            )
            user = (
                "発話:\n" + text + "\n\n"
                "JSONのみを出力:\n{\n  \"anomalies\": [ {\"label\": string, \"kind\": \"PERSON|ORG|EVENT|ENTITY|CLAIM|OTHER\", \"reason\": string} ]\n}"
            )
            resp = self.client.chat(
                model=self.entity_llm_model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                options={"temperature": 0.1, "num_ctx": 4096, "num_batch": 128},
                stream=False,
            )
            content = (resp.get("message") or {}).get("content")
            if not content:
                return None
            s = str(content).strip()
            a = s.find("{")
            b = s.rfind("}")
            if a >= 0 and b > a:
                s = s[a:b+1]
            data = json.loads(s)
            arr = (data or {}).get("anomalies")
            if isinstance(arr, list):
                out: List[Dict[str, Any]] = []
                for it in arr:
                    if not isinstance(it, dict):
                        continue
                    label = (it.get("label") or it.get("name") or "").strip()
                    kind = str(it.get("kind") or it.get("type") or "OTHER").upper()
                    reason = (it.get("reason") or it.get("why") or "").strip()
                    if label:
                        out.append({"label": label, "kind": kind, "reason": reason})
                return out
        except Exception:
            return None
        return None

    def _schedule_soft_ack(self, offender_label: str) -> None:
        remaining = self.soft_ack_max_reminders
        expire = self.turn_counter + self.soft_ack_expire_window
        prev = self.pending_soft_ack.get(offender_label, {})
        # 既存があれば上限で更新
        if prev:
            remaining = max(prev.get("remaining", 0), remaining)
            expire = max(prev.get("expire_turn", 0), expire)
        self.pending_soft_ack[offender_label] = {"remaining": remaining, "expire_turn": expire}

    def _build_entity_correction_plan(self, target_label: str, entity_name_or_list: Any) -> Dict[str, Any]:
        """指摘する側（発言者の反対側）向けの短い指摘スタイル計画を返す。
        entity_name_or_list: str | List[str]
        複数名のときは1つの相づちにまとめて簡潔に触れる。
        """
        if isinstance(entity_name_or_list, list):
            names = [str(n) for n in entity_name_or_list if str(n).strip()]
            if not names:
                names = ["いくつかの名前"]
            if len(names) == 1:
                mention = f"『{names[0]}』"
            elif len(names) == 2:
                mention = f"『{names[0]}』『{names[1]}』"
            else:
                mention = f"『{names[0]}』ほか"
        else:
            mention = f"『{str(entity_name_or_list)}』"
        example = f"{mention}は確認が必要かも"
        plan = {
            "turn_style": {
                "speaker": target_label,
                "length": {"max_chars": 70, "max_sentences": 1},
                "preface": {"aizuchi": True, "aizuchi_list": [example], "prob": 1.0},
                "speech_act": "disagree_short",
                "follow_up": "none",
                "ban": ["praise", "list_format", "long_intro"],
            },
            "cadence": {"avoid_consecutive_monologues": True},
            "closing_hint": "続ける",
        }
        return plan

    def _build_soft_ack_plan(self, offender_label: str) -> Dict[str, Any]:
        """指摘された側向けの“軽い受け流し”トーン指示。複数ターン配布。"""
        # 例フレーズを相づちに埋め込む（実装制約上、自然文をここで示唆）
        examples = ["そうだっけ？", "あれ？違ったかも"]
        plan = {
            "turn_style": {
                "speaker": offender_label,
                "length": {"max_chars": 60, "max_sentences": 1},
                "preface": {"aizuchi": True, "aizuchi_list": examples, "prob": 1.0},
                "speech_act": "agree_short",
                "follow_up": "none",
                "ban": ["praise", "list_format", "long_intro"],
            },
            "cadence": {"avoid_consecutive_monologues": True},
            "closing_hint": "続ける",
        }
        return plan

    # ===== Holistic(文章) レビュー =====
    def _llm_holistic_review_text(self, text: str, context: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """直近発話の整合性・曖昧さ・危うさを文章で簡潔にレビューしてもらう。
        出力はJSONに依存せず、内部用に辞書へ整形して返す（full_textはそのままUI表示可）。
        戻り値例: {"full_text": str, "risk": bool, "risk_level": "low|med|high", "suggestions": [str], "queries": [str]}
        """
        if not self.client:
            return None
        try:
            theme = None
            try:
                # コンテキスト先頭のテーマを拾う（なければNone）
                for m in context:
                    if m.get("role") == "system" and "テーマ" in (m.get("message") or ""):
                        theme = m.get("message")
                        break
            except Exception:
                pass
            participants = [p for p in (self.participants or []) if p]
            part_note = "、".join(participants) if participants else None
            sys_prompt = (
                "あなたは対話ディレクターのレビュアです。以下の日本語発話について、意味の通りに簡潔にレビューし、"
                "曖昧・誤認・場所や時代の取り違え・ありえない主張がないかを人間にわかる文章で短く指摘してください。"
                "次の点を含め、箇条書きではなく2-4文の簡潔な日本語で:")
            if part_note:
                sys_prompt += (
                    f"\n参加者: {part_note}\n"
                    "注意: 上記参加者名やそれらの呼称/あだ名/短縮形は会話上の相手を指すものとして扱い、"
                    "『誰を指すか不明』の指摘対象に含めない。"
                )
            if theme:
                sys_prompt += f"\n会話のテーマ参考: {theme}\n"
            user_prompt = (
                "発話:\n" + text + "\n\n"
                "出力要件: JSON禁止。丁寧語は避け、端的な助言調で。\n"
                "1) まずおかしさがあれば具体的に指摘し、なければ『特に不自然ではない』と述べる。\n"
                "2) 必要なら1つだけ確認質問や検索キーワード案を添える（任意）。")
            resp = self.client.chat(
                model=self.entity_llm_model,
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
                options={"temperature": 0.2, "num_ctx": 4096, "num_batch": 128},
                stream=False,
            )
            content = (resp.get("message") or {}).get("content")
            if not content:
                return None
            full = str(content).strip()
            # リスク推定（簡易）
            risk = any(k in full for k in ["矛盾", "誤り", "誤認", "曖昧", "不自然", "怪しい", "根拠", "ありえない", "辻褄"])
            level = "med" if risk else "low"
            # 追跡クエリ抽出（末尾の疑問文や『〜で検索』等を拾う簡易処理）
            queries: List[str] = []
            try:
                for m in re.finditer(r"[『\"]([^『\"]{4,40})[』\"][でを]?検索", full):
                    queries.append(m.group(1))
            except Exception:
                pass
            suggs: List[str] = []
            # 先頭文をそのまま suggestion にも使う
            first_sent = re.split(r"[。.!?！？]", full)[0].strip()
            if first_sent:
                suggs.append(first_sent)
            return {"full_text": full, "risk": bool(risk), "risk_level": level, "suggestions": suggs, "queries": queries}
        except Exception:
            return None

    def _build_holistic_intervention_text(self, target_label: str, review: Dict[str, Any]) -> Dict[str, Any]:
        """ホリスティックレビュー結果をもとに、短い『ツッコミ+確認』の発話計画を生成。"""
        tip = (review.get("suggestions") or ["それ、確認したいかも"][0]) if review else "それ、確認したいかも"
        q = None
        if review and isinstance(review.get("queries"), list) and review["queries"]:
            q = review["queries"][0]
        aizuchi_list = [tip] if tip else ["それ、確認したいかも"]
        plan = {
            "turn_style": {
                "speaker": target_label,
                "length": {"max_chars": 85, "max_sentences": 2},
                "preface": {"aizuchi": True, "aizuchi_list": aizuchi_list, "prob": 1.0},
                "speech_act": "disagree_short",
                "follow_up": "ask_feel",
                "ban": ["praise", "list_format", "long_intro"],
            },
            "cadence": {"avoid_consecutive_monologues": True},
            "closing_hint": "続ける",
        }
        # 補助: 検索キーワードの示唆を director_debug にも載せる想定だが、ここではplanに含めず最低限
        if q:
            plan["note"] = {"query_suggest": q}
        return plan

    def _build_pushback_plan(self, target_label: str, pushback_text: str) -> Dict[str, Any]:
        """Geminiのpushback短文を使い、そのまま短くツッコミ/訂正要求させる発話計画。"""
        # pushback_text を相づちとして埋め込み、短い反論/確認を促す
        pb = self.auto_repair(pushback_text, max_chars=70)
        if not pb:
            pb = "それ、事実関係あやしい。具体的な根拠ある？"
        plan = {
            "turn_style": {
                "speaker": target_label,
                "length": {"max_chars": 80, "max_sentences": 2},
                "preface": {"aizuchi": True, "aizuchi_list": [pb], "prob": 1.0},
                "speech_act": "disagree_short",
                "follow_up": "ask_feel",
                "ban": ["praise", "list_format", "long_intro"],
            },
            "cadence": {"avoid_consecutive_monologues": True},
            "closing_hint": "続ける",
        }
        return plan

    def _extract_history_metrics(self, dialogue_context: List[Dict[str, Any]]) -> Optional[Dict[str, int]]:
        """履歴に含まれる [history-metrics] ブロックを後方から1つだけ抽出して辞書化。"""
        import re as _re
        try:
            texts = []
            for m in reversed(dialogue_context):
                msg = m.get("message") or ""
                if isinstance(msg, str) and msg.strip():
                    texts.append(msg)
                if len(texts) >= 4:
                    break
            blob = "\n".join(texts)
            start = blob.rfind("[history-metrics]")
            end = blob.rfind("[/history-metrics]")
            if start >= 0 and end > start:
                segment = blob[start:end]
                out = {}
                for k in ("classic_refs", "kyukokumei_refs", "food_refs"):
                    m = _re.search(rf"{k}\s*=\s*(\d+)", segment)
                    if m:
                        out[k] = int(m.group(1))
                return out if out else None
        except Exception:
            return None
        return None

    def _llm_select_search_targets(self, last_text: str, review_text: Optional[str]) -> Optional[List[Dict[str, Any]]]:
        """LLMに『検索すべき対象』を選ばせる。固有名詞(PERSON/ORG/EVENT/ENTITY)と作品(WORK)のみ。
        期待JSON: {"targets":[{"label":str,"kind":"PERSON|ORG|EVENT|ENTITY|WORK"}]}
        """
        if not self.client:
            return None
        try:
            sys = (
                "あなたはレビュー補助です。直近の発話とレビュー文を読み、検索すべき対象だけを選んでください。"
                "一般語や説明語は除外し、固有名詞(PERSON/ORG/EVENT/ENTITY)と、引用/作品/書名など( WORK )に限定。"
                "3件以内。JSONのみ。"
            )
            user = (
                "発話:\n" + (last_text or "") + "\n\n"
                "レビュー:\n" + (review_text or "") + "\n\n"
                "出力:\n{\n  \"targets\": [ {\"label\": string, \"kind\": \"PERSON|ORG|EVENT|ENTITY|WORK\"} ]\n}"
            )
            resp = self.client.chat(
                model=self.entity_llm_model,
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
                options={"temperature": 0, "num_ctx": 4096, "num_batch": 128},
                stream=False,
            )
            content = (resp.get("message") or {}).get("content")
            if not content:
                return None
            s = str(content).strip()
            a = s.find("{"); b = s.rfind("}")
            if a >= 0 and b > a:
                s = s[a:b+1]
            data = json.loads(s)
            arr = (data or {}).get("targets")
            if isinstance(arr, list):
                out: List[Dict[str, Any]] = []
                for it in arr[:3]:
                    if isinstance(it, dict) and (it.get("label") or "").strip():
                        out.append({
                            "label": (it.get("label") or "").strip(),
                            "kind": str(it.get("kind") or "ENTITY").upper()
                        })
                return out
        except Exception:
            return None
        return None

    def _scale_plan_length(self, plan: Dict[str, Any], factor: int = 3, cap: int = 260) -> Dict[str, Any]:
        """turn_style.length.max_chars を係数で拡張（上限cap）。失敗時はそのまま返す。"""
        try:
            ts = plan.get("turn_style", {})
            length = ts.get("length", {})
            max_chars = int(length.get("max_chars")) if length.get("max_chars") is not None else None
            if max_chars:
                new_len = max_chars * max(1, int(factor))
                length["max_chars"] = min(cap, int(new_len))
                ts["length"] = length
                plan["turn_style"] = ts
        except Exception:
            return plan
        return plan

    # ===== Judgeユーティリティ(任意) =====
    @staticmethod
    def judge_text(text: str, max_chars: int = TURN_CHAR_MAX, max_sentences: int = MAX_SENTENCES) -> Dict[str, Any]:
        violations = []
        # 長さ
        if len(text) > max_chars:
            violations.append("too_long")
        # 文数
        sentences = re.split(r"[。.!?！？]+", text.strip())
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

    # （重複定義のクリーンアップ済み。以降は auto_repair のみ保持）

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
