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
    "会話", "議題", "話題", "テーマ", "意見", "考え", "気持ち",
])

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

        # --- 介入拡張: 固有名詞チェックとSoft Ack（軽い受け流し）状態 ---
        # Web検索（MCP）利用の有無（導入済み: Wikipedia RESTを用いた簡易検証）
        self.use_mcp_search = True
        self.mcp_adapter = MCPWebSearchAdapter(language="ja")

        # 指摘された側への「軽い受け流し」継続指示
        self.soft_ack_enabled = True
        self.soft_ack_max_reminders = 2
        self.soft_ack_expire_window = 4  # 現在ターンからの有効ターン幅

        # セッション内キャッシュ/スケジューラ
        self.entity_cache = {}  # name -> {verdict, ts}
        self.pending_soft_ack = {}  # label('A'|'B')-> {remaining, expire_turn}

        # LLMベース抽出の利用設定
        self.use_llm_entity_fallback = True
        self.entity_llm_model = self.model_name
        # ユーザー要望: アルゴ検出に頼らず LLM 抽出も常時併用する
        self.always_llm_entity_scan = True
        # ヒューリスティクス検出は無効化（LLM優先）
        self.use_heuristics = False

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
        if stats["avg_chars_last3"] > 110:
            return random.choice(["reflect", "agree_short", "handoff"])
        return random.choice(["answer", "reflect", "agree_short", "disagree_short"])

    def _decide_max_chars(self, stats: Dict[str, Any]) -> int:
        # メリハリ: たまにロングターンを許可して深掘りを促す
        long_turn = random.random() < 0.22  # 22% でロング
        if long_turn and stats["question_ratio"] <= QUESTION_RATIO_TARGET[1]:
            # 長め: 120〜160
            return random.choice([130, 140, 150, 160])
        # 直近が長すぎる時は抑制
        if stats["avg_chars_last3"] > 120:
            return 85
        # 短すぎる時は増量
        if stats["avg_chars_last3"] < 45:
            return 110
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

    # ===== 参考: 既存 evaluate_dialogue の互換API(機能拡張版) =====
    async def evaluate_dialogue(self, dialogue_context: List[Dict[str, Any]]) -> Dict[str, Any]:
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
        debug_info: Dict[str, Any] = {
            "heuristic_entities": [],
            "llm_entities": [],
            "selected_candidate": None,
            "verification": None,
        }

        # 直近の非Director発話を取得
        last_entry = None
        for m in reversed(dialogue_context):
            if m.get("speaker") and m.get("speaker") != "Director":
                last_entry = m
                break

        # 参加者A/Bの特定を更新（_analyzeはplan_next_turn内でも呼ばれているが安全のため）
        _ = self._analyze(dialogue_context)

        # 1) 固有名詞チェック → 訂正促し（指摘するのは「発言者と別のAI」= 直近発話者の反対側）
        if last_entry and isinstance(last_entry.get("message"), str):
            offender_disp = last_entry.get("speaker") or ""
            offender_label = self._label_of(offender_disp)
            target_label = self._other(offender_label)
            text = last_entry.get("message", "")

            # 1) ヒューリスティクス抽出（無効化設定ならスキップ）
            entities: List[Dict[str, Any]] = []
            if getattr(self, "use_heuristics", True):
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
            for src in (llm_entities if not getattr(self, "use_heuristics", True) else (entities + llm_entities)):
                n = src.get("name")
                if n and n not in seen:
                    seen.add(n)
                    merged.append(src)

            # 4) 候補選定: LLM抽出を優先（PERSON → その他未知語）。ヒューリスティクスは原則不使用
            candidate = None
            # 共通フィルタ: 一般語/略語を除外
            def _passes(name: str) -> bool:
                if not name:
                    return False
                n = name.strip()
                # 自他の呼称（参加者名やその敬称付き）は除外
                if self._is_participant_name(n):
                    return False
                if n in STOP_ENTITY_COMMON:
                    return False
                # 2〜3文字の全大文字略語は除外（例: AI, IT, OS）
                import re as _re
                if _re.fullmatch(r"[A-Z]{1,3}", n):
                    return False
                # 架空キャラクター/ニックネーム的な終端や表現は除外
                if _re.search(r"(ちゃん|たん|っち|にゃん)$", n):
                    return False
                # 明示的なフィクション/キャラ系キーワードを含む場合は除外
                fiction_keywords = [
                    "キャラ", "キャラクター", "VTuber", "ゆるキャラ", "マスコット", "アバター",
                    "二次元", "三次元化", "推し", "擬人化",
                ]
                if any(kw in n for kw in fiction_keywords):
                    return False
                # 英文氏名や複合語は許可
                if " " in n:
                    return True
                # CJK名詞は2文字以上を許可、それ以外は3文字以上
                if _re.search(r"[\u4E00-\u9FFF]", n):
                    return len(n) >= 2
                return len(n) >= 3

            if llm_entities:
                # PERSON優先
                for e in llm_entities:
                    if (e.get("type") or "").upper() == "PERSON" and _passes((e.get("name") or "").strip()):
                        candidate = e
                        break
                if not candidate:
                    for e in llm_entities:
                        n = (e.get("name") or "").strip()
                        if _passes(n):
                            candidate = e
                            break
                if not candidate:
                    candidate = llm_entities[0]
            if not candidate and merged and getattr(self, "use_heuristics", True):
                for e in merged:
                    if (e.get("type") or "").upper() == "PERSON" and _passes((e.get("name") or "").strip()):
                        candidate = e
                        break
                if not candidate:
                    for e in merged:
                        n = (e.get("name") or "").strip()
                        if _passes(n):
                            candidate = e
                            break
                if not candidate:
                    candidate = merged[0]

            if candidate and candidate.get("name"):
                name = candidate.get("name")
                etype = candidate.get("type", "ENTITY")
                debug_info["selected_candidate"] = {"name": name, "type": etype}
                logger.info(f"MCP verify check target='{name}' type={etype}")
                verdict = self._verify_entity(name, etype)
                evidence = None
                ec = self.entity_cache.get(name)
                if isinstance(ec, dict):
                    evidence = ec.get("evidence")
                debug_info["verification"] = {"verdict": verdict, "evidence": evidence}
                if verdict in ("AMBIGUOUS", "NG"):
                    plan = self._build_entity_correction_plan(target_label, name)
                    if self.soft_ack_enabled:
                        self._schedule_soft_ack(offender_label)
                    return {
                        "intervention_needed": True,
                        "reason": "entity_check",
                        "intervention_type": "entity_correction",
                        "message": json.dumps(plan, ensure_ascii=False),
                        "response_length_guide": "簡潔",
                        "confidence": 0.75,
                        "director_debug": debug_info,
                    }

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
        return {
            "intervention_needed": True,
            "reason": "rhythm_control",
            "intervention_type": "length_tempo_speech_act",
            "message": json.dumps(cmd_rhythm, ensure_ascii=False),
            "response_length_guide": "簡潔",
            "confidence": 0.8,
            "director_debug": debug_info,
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
        return unique[:3]

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

        if self.use_mcp_search and self.mcp_adapter:
            try:
                v, url = self.mcp_adapter.verify_entity(name, etype)
                # アダプタの結果を標準化
                if v in ("VERIFIED", "AMBIGUOUS", "NG"):
                    verdict = v
                    evidence = url
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
                options={"temperature": 0.1},
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

    def _schedule_soft_ack(self, offender_label: str) -> None:
        remaining = self.soft_ack_max_reminders
        expire = self.turn_counter + self.soft_ack_expire_window
        prev = self.pending_soft_ack.get(offender_label, {})
        # 既存があれば上限で更新
        if prev:
            remaining = max(prev.get("remaining", 0), remaining)
            expire = max(prev.get("expire_turn", 0), expire)
        self.pending_soft_ack[offender_label] = {"remaining": remaining, "expire_turn": expire}

    def _build_entity_correction_plan(self, target_label: str, entity_name: str) -> Dict[str, Any]:
        """指摘する側（発言者の反対側）向けの短い指摘スタイル計画を返す。"""
        # 軽い相づち例に「指摘のニュアンス」を埋め込む（UI側で例が表示される）
        example = f"『{entity_name}』は確認が必要かも"
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
