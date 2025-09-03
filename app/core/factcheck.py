"""
Fact-checking pipeline (CoVe-inspired):
- Claim extraction (LLM@Ollama, e.g., gemma3)
- Evidence collection (Wikipedia via MCPWebSearchAdapter; optional Tavily API if available)
- FEVER judgment (LLM@Ollama) with quote+URL
- Optional self-consistency (multiple judgments, disagreement rate)

Design goals:
- Zero hard dependency on MCP SDK; pluggable evidence sources using lightweight HTTP
- Safe defaults: if network or API keys are absent, gracefully degrade to Wikipedia-only or skip
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import os
import json
import logging

from .search_adapter import MCPWebSearchAdapter

logger = logging.getLogger(__name__)


@dataclass
class Claim:
    id: str
    text: str
    query: Optional[str] = None


@dataclass
class Evidence:
    source: str  # 'wikipedia' | 'tavily'
    url: Optional[str]
    title: Optional[str] = None
    published: Optional[str] = None
    excerpt: Optional[str] = None


@dataclass
class Judgment:
    claim_id: str
    label: str  # Supported | Refuted | NotEnoughInfo
    quote: Optional[str]
    url: Optional[str]
    rationale: Optional[str] = None


class ClaimExtractor:
    def __init__(self, ollama_client, model: str = "gemma3:4b", temperature: float = 0.1):
        self.client = ollama_client
        self.model = model
        self.temperature = temperature

    def extract(self, text: str, max_items: int = 4) -> List[Claim]:
        if not self.client or not isinstance(text, str) or len(text.strip()) < 20:
            return []
        system = (
            "あなたはファクトチェッカー。入力から検証すべき短い主張だけを抽出し、"
            "最後に各主張の検証クエリ（誰・いつ・どこ・いくつ 等の欠落スロットを補う）を作る。"
        )
        user = (
            "出力はJSONのみ。フォーマット:\n"
            '{"claims":[{"id":"C1","text":"...","query":"..."}]}'
            f"\n入力:\n{text}"
        )
        try:
            resp = self.client.chat(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                options={"temperature": self.temperature, "num_ctx": 4096},
                stream=False,
            )
            raw = (resp.get("message") or {}).get("content") or ""
            s = raw[raw.find("{") : raw.rfind("}") + 1] if "{" in raw and "}" in raw else raw
            data = json.loads(s)
            arr = (data or {}).get("claims") or []
            out: List[Claim] = []
            for i, c in enumerate(arr[: max_items], 1):
                t = (c.get("text") or "").strip()
                q = (c.get("query") or "").strip() or None
                if len(t) >= 8:
                    out.append(Claim(id=f"C{i}", text=t, query=q))
            return out
        except Exception:
            return []


class EvidenceCollector:
    def __init__(self, language: str = "ja"):
        self.wiki = MCPWebSearchAdapter(language=language)
        self._tavily_key = os.getenv("TAVILY_API_KEY")

    def collect(self, claim: Claim, max_items: int = 3) -> List[Evidence]:
        evidences: List[Evidence] = []
        # 1) Wikipedia snippets
        try:
            q = claim.query or claim.text
            snips = self.wiki.search_snippets(q, limit=max_items) or []
            for s in snips:
                evidences.append(
                    Evidence(
                        source="wikipedia",
                        url=s.get("url"),
                        title=s.get("title"),
                        published=None,
                        excerpt=s.get("excerpt"),
                    )
                )
        except Exception:
            pass
        # 2) Tavily (optional)
        if self._tavily_key:
            try:
                import httpx

                payload = {
                    "api_key": self._tavily_key,
                    "query": claim.query or claim.text,
                    "search_depth": "advanced",
                    "include_answers": False,
                    "max_results": max(1, max_items),
                }
                with httpx.Client(timeout=6.0) as client:
                    r = client.post("https://api.tavily.com/search", json=payload)
                    if r.status_code == 200:
                        data = r.json() or {}
                        for it in (data.get("results") or [])[:max_items]:
                            evidences.append(
                                Evidence(
                                    source="tavily",
                                    url=it.get("url"),
                                    title=it.get("title"),
                                    published=it.get("published_date") or it.get("date"),
                                    excerpt=it.get("content"),
                                )
                            )
            except Exception:
                pass
        return evidences[:max_items]


class FeverJudge:
    def __init__(self, ollama_client, model: str = "gemma3:4b", temperature: float = 0.2):
        self.client = ollama_client
        self.model = model
        self.temperature = temperature

    def judge(self, claim: Claim, evidences: List[Evidence]) -> Judgment:
        if not self.client:
            return Judgment(claim_id=claim.id, label="NotEnoughInfo", quote=None, url=None)
        ev_items = [
            {
                "source": e.source,
                "url": e.url,
                "title": e.title,
                "published": e.published,
                "excerpt": (e.excerpt or "")[:800],
            }
            for e in evidences
            if e.url or e.excerpt
        ]
        system = (
            "あなたは事実検証者。各主張に対して与えられた根拠を用い、"
            "FEVER形式のラベル付けを行い、引用(quote)とURLを1つ必ず提示する。"
        )
        user = (
            "出力はJSONのみ。フォーマット:\n"
            '{"label":"Supported|Refuted|NotEnoughInfo","quote":"...","url":"...","rationale":"..."}'
            f"\n主張:\n{claim.text}\n根拠:\n{json.dumps(ev_items, ensure_ascii=False)}"
        )
        try:
            resp = self.client.chat(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                options={"temperature": self.temperature, "num_ctx": 4096},
                stream=False,
            )
            raw = (resp.get("message") or {}).get("content") or ""
            s = raw[raw.find("{") : raw.rfind("}") + 1] if "{" in raw and "}" in raw else raw
            data = json.loads(s)
            label = str(data.get("label") or "NotEnoughInfo").strip()
            quote = (data.get("quote") or None)
            url = (data.get("url") or None)
            rationale = (data.get("rationale") or None)
            return Judgment(claim_id=claim.id, label=label, quote=quote, url=url, rationale=rationale)
        except Exception:
            return Judgment(claim_id=claim.id, label="NotEnoughInfo", quote=None, url=None)


class SelfConsistencyChecker:
    def __init__(self, judge: FeverJudge, rounds: int = 3, temperature: float = 0.7):
        self.judge = judge
        self.rounds = rounds
        self.temperature = temperature

    def check(self, claim: Claim, evidences: List[Evidence]) -> Dict[str, Any]:
        labels: List[str] = []
        if not self.judge or not getattr(self.judge, "client", None):
            return {"rounds": 0, "disagreement_rate": None, "labels": []}
        # temporarily raise temperature
        orig = self.judge.temperature
        try:
            self.judge.temperature = self.temperature
            for _ in range(max(1, self.rounds)):
                j = self.judge.judge(claim, evidences)
                labels.append(j.label)
        finally:
            self.judge.temperature = orig
        if not labels:
            return {"rounds": 0, "disagreement_rate": None, "labels": []}
        maj = max(set(labels), key=labels.count)
        disagree = sum(1 for x in labels if x != maj)
        rate = disagree / len(labels)
        return {"rounds": len(labels), "disagreement_rate": rate, "labels": labels, "majority": maj}


class FactCheckPipeline:
    def __init__(self, ollama_client, model_extract: str = "gemma3:4b", model_judge: str = "gemma3:4b", language: str = "ja"):
        self.extractor = ClaimExtractor(ollama_client, model_extract)
        self.collector = EvidenceCollector(language=language)
        self.judge = FeverJudge(ollama_client, model_judge)
        self.selfcheck = SelfConsistencyChecker(self.judge)

    def run(self, text: str, max_claims: int = 3) -> Dict[str, Any]:
        claims = self.extractor.extract(text, max_items=max_claims)
        results: List[Dict[str, Any]] = []
        for c in claims:
            ev = self.collector.collect(c, max_items=3)
            j = self.judge.judge(c, ev)
            sc = self.selfcheck.check(c, ev)
            results.append(
                {
                    "claim": {"id": c.id, "text": c.text, "query": c.query},
                    "evidence": [e.__dict__ for e in ev],
                    "judgment": j.__dict__,
                    "self_consistency": sc,
                }
            )
        return {"claims": len(claims), "results": results}
