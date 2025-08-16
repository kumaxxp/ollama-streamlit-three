"""
Search/MCP adapter: Minimal web-backed verification helper.
- Primary: Wikipedia (ja) REST API
- Goal: verify whether a named entity (esp. PERSON) likely exists/is notable
- Returns: verdict ('VERIFIED' | 'AMBIGUOUS' | 'NG'), evidence (url or None)

This is a lightweight adapter to serve as an MCP/Web search stand-in.
If a proper MCP client is available later, swap the implementation here.
"""
from __future__ import annotations
import os
from typing import Optional, Tuple

try:
    import httpx  # lightweight HTTP client
except Exception:  # pragma: no cover
    httpx = None  # type: ignore


class MCPWebSearchAdapter:
    def __init__(self, language: str = "ja", timeout: float = 4.0):
        self.lang = language
        self.timeout = timeout

    def verify_entity(self, name: str, etype: str = "PERSON") -> Tuple[str, Optional[str]]:
        """
        Verify entity by checking Wikipedia (ja) summary.
        - Exact summary hit -> 'VERIFIED'
        - Disambiguation or multiple vague hits -> 'AMBIGUOUS'
        - No hits -> 'NG'
        Returns: (verdict, evidence_url)
        """
        v, url, _ = self.verify_entity_detail(name, etype)
        return (v, url)

    def verify_entity_detail(self, name: str, etype: str = "PERSON") -> Tuple[str, Optional[str], Optional[str]]:
        """
        詳細版: (verdict, evidence_url, evidence_text[extract]) を返す。
        """
        if not httpx:
            return ("AMBIGUOUS", None, None)
        q = (name or "").strip()
        if not q:
            return ("AMBIGUOUS", None, None)
        base = f"https://{self.lang}.wikipedia.org"
        # 1) Try exact title summary
        summary_url = f"{base}/api/rest_v1/page/summary/{httpx.URL(q).raw_path.decode('utf-8')}"
        verdict, url, extract = self._fetch_summary_with_text(summary_url)
        if verdict:
            return (verdict, url, extract)
        # 2) Fallback: title search
        search_url = f"{base}/w/rest.php/v1/search/title?q={httpx.QueryParams({'q': q, 'limit': 3})['q']}&limit=3"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.get(search_url, headers={"accept": "application/json"})
                if r.status_code != 200:
                    return ("AMBIGUOUS", None, None)
                data = r.json()
                pages = (data or {}).get("pages") or []
                candidate = None
                for p in pages:
                    title = (p.get("title") or "").strip()
                    if title == q:
                        candidate = title
                        break
                if not candidate and pages:
                    candidate = (pages[0].get("title") or "").strip()
                if candidate:
                    s_url = f"{base}/api/rest_v1/page/summary/{httpx.URL(candidate).raw_path.decode('utf-8')}"
                    v2, u2, ex2 = self._fetch_summary_with_text(s_url)
                    return (v2 or "AMBIGUOUS", u2, ex2)
        except Exception:
            return ("AMBIGUOUS", None, None)
        return ("AMBIGUOUS", None, None)

    def _fetch_summary(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.get(url, headers={"accept": "application/json"})
                if r.status_code == 404:
                    return (None, None)
                if r.status_code != 200:
                    return (None, None)
                data = r.json()
                ptype = data.get("type")
                content_urls = ((data.get("content_urls") or {}).get("desktop") or {})
                page_url = content_urls.get("page")
                if ptype == "standard":
                    return ("VERIFIED", page_url)
                if ptype == "disambiguation":
                    return ("AMBIGUOUS", page_url)
                # other types -> ambiguous
                return ("AMBIGUOUS", page_url)
        except Exception:
            return (None, None)

    def _fetch_summary_with_text(self, url: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.get(url, headers={"accept": "application/json"})
                if r.status_code == 404:
                    return (None, None, None)
                if r.status_code != 200:
                    return (None, None, None)
                data = r.json()
                ptype = data.get("type")
                content_urls = ((data.get("content_urls") or {}).get("desktop") or {})
                page_url = content_urls.get("page")
                extract = data.get("extract")
                if ptype == "standard":
                    return ("VERIFIED", page_url, extract)
                if ptype == "disambiguation":
                    return ("AMBIGUOUS", page_url, extract)
                return ("AMBIGUOUS", page_url, extract)
        except Exception:
            return (None, None, None)
