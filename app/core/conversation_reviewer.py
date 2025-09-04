"""
Conversation Reviewer (Holistic)
現在までの会話履歴を読み、テーマ順守・建設性・進捗・改善提案を短く日本語で評価する。
出力はプレーンテキスト（UIでそのまま表示）。
"""
from __future__ import annotations
from typing import List, Dict, Any


def _build_prompt(history: List[Dict[str, str]], theme: str | None = None) -> str:
    lines: List[str] = []
    if theme:
        lines.append(f"【テーマ】{theme}")
    lines.append("【これまでの会話（新しい順ではなく時系列・最大10件）】")
    for h in history[-10:]:
        sp = str(h.get("speaker", "?"))
        msg = str(h.get("message", "")).strip()
        if not msg:
            continue
        if len(msg) > 400:
            msg = msg[:400] + "…"
        lines.append(f"- {sp}: {msg}")

    instruction = (
        "あなたは日本語の会話レビュアです。上の会話全体を短く評価してください。\n"
        "- 要約（2〜3文）\n"
        "- 良かった点（1〜2点）\n"
        "- 改善点（1〜2点）\n"
        "- 次の一手（1文の提案）\n"
        "出力は日本語のプレーンテキストのみ。見出しに絵文字や記号を多用しない。箇条書きは最大3行まで。\n"
    )
    return instruction + "\n\n" + "\n".join(lines)


def review_conversation_text(
    history: List[Dict[str, str]],
    *,
    ollama_client,
    model_name: str = "gemma3:4b",
    temperature: float = 0.2,
    theme: str | None = None,
) -> str:
    try:
        if ollama_client is None:
            raise RuntimeError("ollama client not available")
        prompt = _build_prompt(history, theme)
        resp = ollama_client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a concise Japanese dialogue reviewer. Output plain Japanese text only."},
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": float(temperature),
                "num_ctx": 2048,
                "num_predict": 280,
            },
        )
        return (resp.get("message", {}) or {}).get("content", "") or ""
    except Exception:
        # フォールバック（簡易ひな形）
        return (
            "【要約】対話は概ねテーマに沿って進行しています。\n"
            "【良かった点】相互に相手へ反応しており、論点が絞られつつあります。\n"
            "【改善点】主張の根拠や具体例を一つに絞って短く補強すると明確になります。\n"
            "【次の一手】相手が答えやすい一問を添えて、論点を一歩前に進めましょう。"
        )
