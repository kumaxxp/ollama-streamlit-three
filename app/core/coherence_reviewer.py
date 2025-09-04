"""
Coherence Reviewer
対話履歴と最新発話の整合性/矛盾/破綻をLLMで素早くチェックする軽量レビュー。

依存: ollama.Client（呼び出し元から渡す）。
出力: 小さめのJSON（UIで表示しやすい）。
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional
import json


def _build_prompt(history: List[Dict[str, str]], latest: str) -> str:
    """
    history: [{speaker, message}], 最新の latest を含む短いレビュー用テキストを構築
    """
    lines: List[str] = []
    lines.append("【直近の対話】")
    for h in history[-6:]:
        sp = str(h.get("speaker", "?"))
        msg = str(h.get("message", "")).strip()
        if not msg:
            continue
        # 長すぎる行は切り詰め（モデルへの負荷軽減）
        if len(msg) > 500:
            msg = msg[:500] + "…"
        lines.append(f"- {sp}: {msg}")
    lines.append("\n【レビュー対象（最新発話）】\n" + (latest[:800] + ("…" if len(latest) > 800 else "")))

    instruction = (
        "あなたは会話の監査AIです。最新発話が直前までの会話と比べて、破綻・矛盾・論理飛躍・唐突な話題転換がないかを厳密かつ短時間で点検してください。\n"
        "必ず次のJSON形式のみで出力してください。日本語で簡潔に。\n"
        "{\n"
        "  \"coherence_score\": 0-100の整数,\n"
        "  \"has_issues\": true|false,\n"
        "  \"summary\": \"全体所見（短く）\",\n"
        "  \"issues\": [ {\n"
        "     \"type\": \"contradiction|incoherence|topic_shift|tone_mismatch\",\n"
        "     \"excerpt\": \"問題箇所の抜粋（20〜80字）\",\n"
        "     \"explain\": \"何が問題か（短く）\"\n"
        "  } ],\n"
        "  \"suggest_fix\": \"修正提案（任意・1文）\"\n"
        "}\n"
        "注意: JSON以外の文字を書かない。要素は存在しない場合でも空配列/空文字で埋める。\n"
    )
    return instruction + "\n\n" + "\n".join(lines)


def review_coherence(
    history: List[Dict[str, str]],
    latest_message: str,
    *,
    ollama_client,
    model_name: str = "gemma3:4b",
    temperature: float = 0.1,
) -> Dict[str, Any]:
    """LLMで簡易整合性レビューを実行し、JSON辞書で返す。
    ollama_client が無い場合や失敗時はフォールバックの簡易結果を返す。
    """
    try:
        if ollama_client is None:
            raise RuntimeError("ollama client not available")

        prompt = _build_prompt(history, latest_message or "")
        resp = ollama_client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a strict but concise Japanese conversation consistency checker. Output only JSON."},
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": float(temperature),
                "num_ctx": 2048,
                "num_predict": 256,
            },
        )
        text = (resp.get("message", {}) or {}).get("content", "")
        data = json.loads(text)
        # 最低限の正規化
        return {
            "coherence_score": int(max(0, min(100, int(data.get("coherence_score", 70))))),
            "has_issues": bool(data.get("has_issues", False)),
            "summary": str(data.get("summary", ""))[:300],
            "issues": data.get("issues", []) or [],
            "suggest_fix": str(data.get("suggest_fix", ""))[:200],
        }
    except Exception:
        # フォールバック（簡易ヒューリスティック）
        score = 80
        issues: List[Dict[str, str]] = []
        has_q = any(x in (latest_message or "") for x in ["？", "?"])
        if not latest_message:
            score = 40
            issues.append({
                "type": "incoherence",
                "excerpt": "(空の応答)",
                "explain": "内容が空です",
            })
        elif len(latest_message) > 800:
            score = 65
            issues.append({
                "type": "incoherence",
                "excerpt": latest_message[:60] + "…",
                "explain": "冗長の可能性（非常に長い）",
            })
        return {
            "coherence_score": score,
            "has_issues": bool(issues),
            "summary": "簡易チェック（フォールバック）",
            "issues": issues,
            "suggest_fix": "要点を1つに絞って短く述べると一貫性が上がります。" if issues else "",
        }
