"""
Dialogue Controller - ビジネスロジック統合モジュール
UIから完全に独立した対話制御ロジック
"""
import json
import time
from typing import Dict, Any, List, Optional, Generator, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# 正しいクラス名でインポート
from .director import AutonomousDirector
from .agent import Agent
from .dialogue_manager import DialogueManager


@dataclass
class DialogueConfig:
    """対話設定"""
    theme: str
    agent1_name: str
    agent2_name: str
    max_turns: int = 10
    director_config: Dict[str, Any] = field(default_factory=dict)
    model_params: Dict[str, Any] = field(default_factory=dict)
    auto_mode: bool = True


@dataclass
class DialogueState:
    """対話状態"""
    turn_count: int = 0
    phase: str = "exploration"
    total_tokens: int = 0
    start_time: float = field(default_factory=time.time)
    history: List[Dict] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    is_paused: bool = False


class DialogueController:
    """
    対話制御の中核ロジック
    UIから完全に独立して動作
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.config_path = Path(config_path) if config_path else None
        self.director = None
        self.dialogue_manager = None
        self.agents = {}
        self.state = None
        self.config = None
        
        # Ollamaクライアントを初期化
        try:
            import ollama
            self.ollama_client = ollama.Client()
        except Exception as e:
            print(f"Warning: Ollama client initialization failed: {e}")
            self.ollama_client = None
        
        # 設定をロード
        if self.config_path and self.config_path.exists():
            self._load_config()
    
    def _load_config(self) -> None:
        """設定ファイルをロード"""
        with open(self.config_path) as f:
            self.base_config = json.load(f)
    
    def initialize_session(self, config: DialogueConfig) -> None:
        """
        セッションを初期化
        
        Args:
            config: 対話設定
        """
        self.config = config
        self.state = DialogueState()
        
        # コンポーネントを初期化
        self._initialize_components()
        
        # 初期コンテキストを設定
        self._setup_initial_context()
    
    def _initialize_components(self) -> None:
        """コンポーネントを初期化"""
        # AutonomousDirectorを初期化（正しいクラス名）
        director_model = self.config.director_config.get('model', 'gemma3:4b')
        self.director = AutonomousDirector(
            ollama_client=self.ollama_client,
            model_name=director_model
        )
        # Directorへセッションテーマを共有（オフトピック検知用）
        try:
            if hasattr(self.director, 'session_theme'):
                self.director.session_theme = self.config.theme
        except Exception:
            pass
        
        # DialogueManagerを初期化
        self.dialogue_manager = DialogueManager(
            ollama_client=self.ollama_client,
            director_model=director_model
        )
        
        # テーマを設定
        self.dialogue_manager.theme = self.config.theme

        # エージェントを初期化
        agent_model = self.config.model_params.get('model', 'qwen2.5:7b')
        # 個別温度（指定がなければ共通temperature/既定値にフォールバック）
        agent1_temp = self.config.model_params.get('agent1_temperature', self.config.model_params.get('temperature', 0.7))
        agent2_temp = self.config.model_params.get('agent2_temperature', self.config.model_params.get('temperature', 0.7))

        self.agents = {
            self.config.agent1_name: Agent(
                agent_id="agent1",
                character_type=self.config.agent1_name,
                model_name=agent_model,
                temperature=agent1_temp,
                ollama_client=self.ollama_client
            ),
            self.config.agent2_name: Agent(
                agent_id="agent2",
                character_type=self.config.agent2_name,
                model_name=agent_model,
                temperature=agent2_temp,
                ollama_client=self.ollama_client
            )
        }
    
    def _setup_initial_context(self) -> None:
        """初期コンテキストを設定"""
        # 各エージェントに初期設定
        for agent_name, agent in self.agents.items():
            agent.set_session_context(
                theme=self.config.theme,
                goal="建設的な議論",
                phase="exploration"
            )
    
    def run_turn(self) -> Generator[Dict[str, Any], None, None]:
        """
        1ターンを実行（ジェネレータ）
        
        Yields:
            イベント辞書 {"type": str, "data": Any}
        """
        if not self.state.is_active or self.state.is_paused:
            return
        
        self.state.turn_count += 1
        turn_start_time = time.time()
        
        # 現在の話者を決定
        current_speaker = self._get_current_speaker()
        current_agent = self.agents[current_speaker]
        
        # Director分析を実行（必要に応じて）
        if self._should_analyze():
            analysis = self._perform_analysis()
            yield {"type": "director_analysis", "data": analysis}
            
            # 介入が必要な場合
            if analysis.get("intervention_needed"):
                intervention = self._generate_intervention(analysis)
                # デバッグ情報を透過
                if isinstance(analysis, dict) and analysis.get("director_debug"):
                    intervention["director_debug"] = analysis.get("director_debug")
                # review_directives があれば保持（TTL付き）
                if isinstance(analysis, dict) and analysis.get("review_directives"):
                    rd = analysis.get("review_directives")
                    if isinstance(rd, dict):
                        ttl = int(rd.get("ttl_turns", 2))
                        rd_entry = {"data": rd, "ttl": max(1, ttl)}
                        # 上書き（常に最新を優先）
                        self._active_review_directives = rd_entry
                yield {"type": "director_intervention", "data": intervention}
                
                # エージェントへの指示を更新
                self._update_agent_instructions(intervention)
                # 検出/検証情報を次のプロンプト生成に利用できるよう保存
                self._stash_director_findings(intervention)
        
    # エージェントの応答を生成
        yield {"type": "agent_start", "data": {"agent": current_speaker}}
        
        # 同期的に応答を生成（簡略化のため）
        context = self._build_agent_context(current_speaker)
        # 直近のディレクター検出情報があれば、コンテキストに付与
        findings = getattr(self, "_last_director_findings", None)
        if findings:
            context["director_findings"] = findings
        # review_directives がアクティブなら、Agentのsystem先頭に短いブロックとして反映
        rd_entry = getattr(self, "_active_review_directives", None)
        if isinstance(rd_entry, dict) and isinstance(rd_entry.get("data"), dict) and rd_entry.get("ttl", 0) > 0:
            rd = rd_entry["data"]
            lines = []
            # 小さな追加価値: geo_nudge の定型2種からランダムで指示語をヒント表示
            import random as _rand
            geo_hint = _rand.choice(["そんな場所ありましたっけ？", "それ日本の話？"]) if "geo_nudge" in (rd.get("required_actions") or []) else None
            if geo_hint:
                lines.append(f"- 地理の整合性に短く触れる: {geo_hint}")
            # 強い引用禁止
            if "ban_classic_quotes" in (rd.get("required_actions") or []):
                lines.append("- 古典/名言/物語の引用は禁止。『〜曰く』『〜のように』等は使わない。自分の言葉で述べる。")
            if "summarize_work" in (rd.get("required_actions") or []):
                lines.append("- 古典/作品は要旨を一言で触れる（引用は控えめ）")
            if "cite_one_url_if_available" in (rd.get("required_actions") or []):
                lines.append("- URLがあれば1つだけ添える（任意）")
            if "one_concise_question" in (rd.get("ensure") or []):
                lines.append("- 最後に短い質問を1つだけ")
            avoid = rd.get("avoid") or []
            if "overuse_classics" in avoid:
                lines.append("- 古典の引用は続けて多用しない")
            # 短い“From Director Review”ブロックを findings に差し込んでUserプロンプト側に見えるようにする
            if lines:
                context.setdefault("director_findings", {})
                context["director_findings"]["review_block"] = "\n".join(lines)
                # 元ディレクティブも渡して、Agentがsystemで強制できるようにする
                context["director_findings"]["review_directives"] = rd

        # メトリクスに基づく強制ルール（分析が走っていなくても適用）
        try:
            mb = getattr(self, "_last_metrics_block", None)
            metrics = self._parse_metrics_block(mb) if isinstance(mb, str) else None
            if metrics and int(metrics.get("classic_refs", 0)) >= 3:
                # レビュー文言を追加
                context.setdefault("director_findings", {})
                existing = context["director_findings"].get("review_block") or ""
                add_line = "- 古典/名言/物語の引用は禁止。『〜曰く』『〜のように』等は使わない。自分の言葉で述べる。"
                if add_line not in existing:
                    context["director_findings"]["review_block"] = (existing + ("\n" if existing else "") + add_line)
                # review_directives も合成
                rd2 = context["director_findings"].get("review_directives") or {}
                req = set(rd2.get("required_actions") or [])
                req.add("ban_classic_quotes")
                av = set(rd2.get("avoid") or [])
                av.add("overuse_classics")
                rd2.update({
                    "required_actions": list(req),
                    "avoid": list(av),
                    "ttl_turns": max(2, int((rd2.get("ttl_turns") or 2)))
                })
                context["director_findings"]["review_directives"] = rd2
        except Exception:
            pass

        # 事前にsystem/userプロンプトを生成してUIに渡す
        try:
            agent_obj = current_agent
            system_prompt = agent_obj._build_system_prompt()
            user_prompt = agent_obj.build_prompt(context)
            yield {"type": "agent_prompts", "data": {"agent": current_speaker, "system_prompt": system_prompt, "user_prompt": user_prompt}}
            # findings は1ターンのみ有効にするため使用後クリア
            if hasattr(self, "_last_director_findings"):
                self._last_director_findings = None
            # review_directives のTTLを消費
            if hasattr(self, "_active_review_directives") and isinstance(getattr(self, "_active_review_directives"), dict):
                try:
                    self._active_review_directives["ttl"] -= 1
                    if self._active_review_directives["ttl"] <= 0:
                        self._active_review_directives = None
                except Exception:
                    self._active_review_directives = None
        except Exception:
            # 生成失敗しても続行
            system_prompt = None
            user_prompt = None
        
        try:
            # asyncioを使わずに同期的に処理
            import asyncio
            
            # 非同期関数を同期的に実行
            if asyncio.iscoroutinefunction(current_agent.generate_response):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response = loop.run_until_complete(
                    current_agent.generate_response(context, system_prompt=system_prompt, user_prompt=user_prompt)
                )
                loop.close()
            else:
                # 同期関数の場合
                response = current_agent.generate_response(context, system_prompt=system_prompt, user_prompt=user_prompt)
        except Exception as e:
            print(f"Response generation error: {e}")
            response = "申し訳ございません、応答の生成に失敗しました。"
        
        # 監視LLM: メトリクスはログに保存せず、次処理用に一時保持のみ
        try:
            last5 = []
            if self.state.history:
                for h in self.state.history[-5:]:
                    c = h.get("content", "")
                    last5.append(c.get("message") if isinstance(c, dict) else str(c))
            self._last_metrics_block = self.director.build_history_metrics_block(last5) if hasattr(self.director, 'build_history_metrics_block') else None
        except Exception:
            self._last_metrics_block = None

        # If the agent returned a structured error, preserve it in the event
        if isinstance(response, dict) and response.get('error'):
            yield {"type": "agent_response", "data": {"agent": current_speaker, "response": response.get('message'), "error": True, "detail": response.get('detail')}}
        else:
            yield {"type": "agent_response", "data": {"agent": current_speaker, "response": response}}
        
        # 履歴を更新
        self._update_history(current_speaker, response)
        
        # メトリクスを更新
        turn_time = time.time() - turn_start_time
        self._update_metrics(turn_time, len(response))
        
        yield {"type": "turn_complete", "data": self.get_state_summary()}

    def _stash_director_findings(self, intervention: Dict[str, Any]) -> None:
        """Directorの検出/検証情報を次のプロンプトで使えるよう保持する。"""
        try:
            dbg = intervention.get("director_debug") if isinstance(intervention, dict) else None
            if not isinstance(dbg, dict):
                return
            findings: Dict[str, Any] = {"timestamp": datetime.now().isoformat()}

            # エンティティ検証（従来）
            sel = dbg.get("selected_candidate") or {}
            ver = dbg.get("verification") or {}
            name = sel.get("name")
            if name:
                # evidence_text は長い場合があるため抜粋を作る
                evidence_text = ver.get("evidence_text") if isinstance(ver, dict) else None
                excerpt = None
                if isinstance(evidence_text, str) and evidence_text.strip():
                    txt = evidence_text.strip().replace("\n", " ")
                    excerpt = (txt[:280] + ("…" if len(txt) > 280 else ""))
                findings.update({
                    "entity_name": name,
                    "entity_type": sel.get("type"),
                    "verdict": ver.get("verdict"),
                    "evidence": ver.get("evidence"),
                    "evidence_text": evidence_text,
                    "evidence_excerpt": excerpt,
                })

            # ホリスティックレビュー（文章）
            htext = dbg.get("holistic_text")
            if isinstance(htext, str) and htext.strip():
                findings["holistic_text"] = htext.strip()

            # 追加: その場検索の結果(research)があれば共有
            research = dbg.get("research")
            if isinstance(research, list) and research:
                # 1件目だけを短く格納
                r0 = research[0]
                findings["research"] = {
                    "query": r0.get("query"),
                    "verdict": r0.get("verdict"),
                    "evidence": r0.get("evidence"),
                    "evidence_excerpt": (r0.get("evidence_text") or "")[:280]
                }
            # 作品検証の結果（あればそのまま渡す）
            works = dbg.get("works_detected")
            if isinstance(works, list) and works:
                findings["works_detected"] = works
            # Wikipediaスニペット（簡易検索結果）
            wiki_snips = dbg.get("wiki_snippets")
            if isinstance(wiki_snips, list) and wiki_snips:
                # 多すぎると冗長なので2件に絞る
                findings["wiki_snippets"] = wiki_snips[:2]

            # 何かしらあれば保存
            if len(findings) > 1:  # timestamp 以外がある
                self._last_director_findings = findings
        except Exception:
            # 非致命
            self._last_director_findings = None
    
    def _should_analyze(self) -> bool:
        """分析を実行すべきか判定"""
        # 設定された間隔でチェック
        check_interval = self.config.director_config.get("check_interval", 2)
        return self.state.turn_count % check_interval == 0
    
    def _perform_analysis(self) -> Dict[str, Any]:
        """Director分析を実行（同期版）
        Directorが期待する履歴（speaker/listener/message）形式に整形して渡す。
        """
        recent_history_raw = self.state.history[-4:] if len(self.state.history) >= 4 else self.state.history
        # 2者会話前提でlistenerを相手に設定
        agent_names: List[str] = list(self.agents.keys())
        formatted: List[Dict[str, Any]] = []
        for entry in recent_history_raw:
            role = entry.get("role", "")
            content = entry.get("content", "")
            # content が dict（エラー情報等）の場合は message を抽出
            if isinstance(content, dict):
                message = content.get("message", "")
            else:
                message = str(content)

            speaker_disp = self.agents[role].character.get("name", role) if role in self.agents else role or "不明"
            # 相手（listener）を推定
            if len(agent_names) == 2 and role in agent_names:
                listener_key = agent_names[0] if agent_names[1] == role else agent_names[1]
                listener_disp = self.agents[listener_key].character.get("name", listener_key)
            else:
                listener_disp = "相手"

            formatted.append({
                "speaker": speaker_disp,
                "listener": listener_disp,
                "message": message
            })
        # メトリクスブロックがあれば、ディレクター入力の末尾メッセージに一時的に付与（ログには保存しない）
        try:
            if formatted and isinstance(getattr(self, "_last_metrics_block", None), str) and self._last_metrics_block.strip():
                formatted[-1]["message"] = (formatted[-1]["message"] or "").rstrip() + "\n\n" + self._last_metrics_block
        except Exception:
            pass
        
        # 非同期メソッドを同期的に実行
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.director.evaluate_dialogue(formatted)
            )
            loop.close()
            return result
        except Exception as e:
            print(f"Analysis error: {e}")
            return {"intervention_needed": False}
    
    def _generate_intervention(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """介入を生成"""
        return {
            "type": analysis.get("intervention_type", "none"),
            "message": analysis.get("message", ""),
            "reason": analysis.get("reason", "")
        }
    
    def _update_agent_instructions(self, intervention: Dict[str, Any]) -> None:
        """エージェントへの指示を更新
        - 介入JSON（文字列）を解析し、人が読みやすい日本語の指示へ変換
        - 指定話者（A/B）が含まれていれば、そのエージェントに適用
        """
        raw = intervention.get("message", "")
        plan = None
        if isinstance(raw, str):
            # JSONであれば辞書化
            try:
                plan = json.loads(raw)
            except Exception:
                plan = None
        elif isinstance(raw, dict):
            plan = raw

        # 指示文と注意点を生成
        instruction_text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
        attention_points: List[str] = ["建設的な議論を心がける"]
        target_agent_key = self._get_next_speaker()

        if plan and isinstance(plan, dict):
            ts = plan.get("turn_style", {})
            length = ts.get("length", {})
            preface = ts.get("preface", {})
            bans = ts.get("ban", []) or []
            speech_act = ts.get("speech_act", "")
            follow_up = ts.get("follow_up", "")
            label = ts.get("speaker")

            # 指定話者があれば A/B を実エージェントにマップ
            if label in ("A", "B"):
                mapped = self._map_label_to_agent_key(label)
                if mapped:
                    target_agent_key = mapped

            # 自然言語の指示を合成
            max_chars = length.get("max_chars")
            max_sent = length.get("max_sentences")
            aizuchi_on = preface.get("aizuchi")
            # 防御的にフラット化+文字列化
            raw_aizuchi = preface.get("aizuchi_list") or []
            aizuchi_list: List[str] = []
            try:
                if isinstance(raw_aizuchi, list):
                    for it in raw_aizuchi:
                        if isinstance(it, list):
                            aizuchi_list.extend([str(x) for x in it])
                        else:
                            aizuchi_list.append(str(it))
                elif isinstance(raw_aizuchi, (str, int, float)):
                    aizuchi_list = [str(raw_aizuchi)]
            except Exception:
                # 型不整合時は空扱い
                aizuchi_list = []

            nl_parts: List[str] = []
            # 話法
            if speech_act:
                mapping = {
                    "ask": "短く問いかける",
                    "answer": "端的に答える",
                    "reflect": "相手の趣旨を要約して返す",
                    "agree_short": "一言で軽く同意する",
                    "disagree_short": "穏やかに短く異議を述べる",
                    "handoff": "相手に話を渡す",
                }
                nl_parts.append(f"話法: {mapping.get(speech_act, speech_act)}")
            # 長さ
            if max_chars or max_sent:
                len_text = []
                if max_chars:
                    len_text.append(f"{max_chars}文字以内")
                if max_sent:
                    len_text.append(f"最大{max_sent}文")
                nl_parts.append("応答の長さ: " + "・".join(len_text))
                attention_points.append("簡潔")
            # 相槌
            if aizuchi_on:
                ex = f"（例: {'、'.join(aizuchi_list)}）" if aizuchi_list else ""
                nl_parts.append(f"冒頭に軽い相槌を添えてもよい{ex}")
            # 禁止事項
            if bans:
                jp = {
                    "praise": "過度な称賛",
                    "long_intro": "長い前置き",
                    "list_format": "箇条書き/リスト形式",
                }
                ban_list = [jp.get(b, b) for b in bans]
                nl_parts.append("禁止: " + "・".join(ban_list))
            # 追従
            if follow_up == "none":
                nl_parts.append("今回は質問は付けない")

            instruction_text = "\n".join(nl_parts)

            # review_directives（アクティブ）に基づく強制追記: 引用禁止
            try:
                rd_entry = getattr(self, "_active_review_directives", None)
                rd = rd_entry.get("data") if isinstance(rd_entry, dict) else None
                reqs = (rd or {}).get("required_actions") if isinstance(rd, dict) else []
                avoids = (rd or {}).get("avoid") if isinstance(rd, dict) else []
                forbid_quotes = False
                if isinstance(reqs, list) and "ban_classic_quotes" in reqs:
                    forbid_quotes = True
                if isinstance(avoids, list) and "overuse_classics" in avoids:
                    forbid_quotes = True
                if forbid_quotes:
                    if "引用禁止" not in attention_points:
                        attention_points.append("引用禁止")
                    if "古典/名言/物語の引用は禁止" not in instruction_text:
                        instruction_text = (instruction_text + "\n" if instruction_text else "") + "禁止: 古典/名言/物語の引用（『〜曰く』『〜のように』等）"
            except Exception:
                pass

            # 追加: geoやpushback相当の相槌が含まれる場合、自然言語で強い誘導文を添える
            try:
                dbg = intervention.get("director_debug") if isinstance(intervention, dict) else None
                extra_lines: List[str] = []
                if isinstance(dbg, dict):
                    # geo 情報から日本外の可能性をツッコミ候補に
                    geo = dbg.get("geo") or {}
                    if isinstance(geo, dict) and geo:
                        for k, g in list(geo.items())[:1]:
                            if g and g.get("in_japan") is False:
                                extra_lines.append(f"地理の整合性に注意。『{k}』は日本外の位置です。『そんな場所ありましたっけ？』のニュアンスで短く確認してOK。")
                if aizuchi_on and aizuchi_list:
                    # pushback候補の相槌があるなら、それを冒頭に必ず置くよう明示
                    extra_lines.append("冒頭に上記の短いツッコミ相槌を必ず1つ置いてから、1文だけ確認する。")
                if extra_lines:
                    instruction_text = instruction_text + "\n" + "\n".join(extra_lines)
            except Exception:
                pass

            # 注意点（UI側で拾われる想定）
            if max_chars or max_sent:
                attention_points.append(
                    f"長さ遵守: {(str(max_chars)+'文字以内') if max_chars else ''}{'・' if max_chars and max_sent else ''}{('最大'+str(max_sent)+'文') if max_sent else ''}"
                )
            if bans:
                attention_points.append("禁止事項を守る")
            if aizuchi_on and aizuchi_list:
                attention_points.append(f"相槌可: {aizuchi_list[0]}")

            # 開幕は最長にするオーバーライド（ディレクターの短い指示より優先）
            try:
                if getattr(self.state, 'turn_count', 0) == 1:
                    opening_lines = [
                        "開幕はやや長めに（200-300文字・2〜4文）",
                        "まず導入→具体例や観点を一つ→最後に短く問いを添える",
                        "箇条書き・見出し・絵文字は禁止"
                    ]
                    instruction_text = "\n".join([
                        "話法: 導入を述べてから一つだけ問いを添える",
                        "応答の長さ: 300文字以内・最大4文",
                        *opening_lines
                    ])
                    # 注意点を上書き/強化
                    attention_points = [
                        "開幕のみ長め（200-300字・2〜4文）",
                        "最後に問いは1つだけ",
                        "禁止事項を守る",
                    ]
            except Exception:
                pass

        # 対象エージェントへ適用
        if target_agent_key in self.agents:
            self.agents[target_agent_key].add_directive(
                instruction=instruction_text,
                attention_points=attention_points
            )

    def _map_label_to_agent_key(self, label: str) -> Optional[str]:
        """対話履歴から表示名の出現順を用いて A/B をエージェントキーにマップ"""
        if label not in ("A", "B"):
            return None
        order_display: List[str] = []
        if self.state and self.state.history:
            for h in self.state.history:
                role = h.get("role")
                if role in self.agents:
                    disp = self.agents[role].character.get("name", role)
                    if disp not in order_display:
                        order_display.append(disp)
                if len(order_display) >= 2:
                    break
        # ディスプレイ名→キー
        display_to_key = {self.agents[k].character.get("name", k): k for k in self.agents}
        if not order_display:
            # フォールバック: 登録順で A/B
            keys = list(self.agents.keys())
            return keys[0] if label == "A" else (keys[1] if len(keys) > 1 else keys[0])
        first = order_display[0]
        second = order_display[1] if len(order_display) > 1 else None
        if label == "A":
            return display_to_key.get(first)
        else:
            return display_to_key.get(second, display_to_key.get(first))
    
    def _get_current_speaker(self) -> str:
        """現在の話者を取得"""
        # run_turn 冒頭で turn_count を +1 済みのため、初回は index 0 になるよう -1 で補正
        agent_names = list(self.agents.keys())
        idx = (self.state.turn_count - 1) % 2
        return agent_names[idx]
    
    def _get_next_speaker(self) -> str:
        """次の話者を取得"""
        # 現在 = (turn_count-1)%2 → 次 = turn_count%2
        agent_names = list(self.agents.keys())
        idx = (self.state.turn_count) % 2
        return agent_names[idx]
    
    def _build_agent_context(self, speaker_name: str) -> Dict:
        """エージェント用のコンテキストを構築"""
        # 相手の名前を取得
        agent_names = list(self.agents.keys())
        opponent_name = [n for n in agent_names if n != speaker_name][0]
        
        # 最後の相手の発言を取得
        opponent_message = ""
        if self.state.history:
            for entry in reversed(self.state.history):
                if entry.get("role") == opponent_name:
                    last_content = entry.get("content", "")
                    if isinstance(last_content, dict):
                        opponent_message = last_content.get("message", "")
                    else:
                        opponent_message = str(last_content)
                    break
        # 直近履歴をUI用に整形（speaker/message 形式）
        recent: List[Dict] = []
        if self.state.history:
            import re as _re
            for h in self.state.history[-3:]:
                role = h.get("role", "")
                content = h.get("content", "")
                if isinstance(content, dict):
                    content = content.get("message", "")
                # 表示用に history-metrics ブロックを除去（ログ内容は変更しない）
                msg = str(content)
                msg = _re.sub(r"\[history-metrics\][\s\S]*?\[/history-metrics\]", "", msg).strip()
                display_name = self.agents[role].character.get("name", role) if role in self.agents else role
                recent.append({
                    "speaker": display_name,
                    "message": msg
                })

        # メトリクスはUser Prompt専用で別セクションに出すため、コンテキストに渡すのみ
        if isinstance(getattr(self, "_last_metrics_block", None), str) and self._last_metrics_block and self._last_metrics_block.strip():
            metrics_block = self._last_metrics_block
        else:
            metrics_block = None

        return {
            "opponent_name": self.agents[opponent_name].character.get("name", opponent_name),
            "opponent_message": opponent_message,
            "recent_history": recent,
            **({"metrics_block": metrics_block} if metrics_block else {})
        }

    def _parse_metrics_block(self, block: Optional[str]) -> Optional[Dict[str, int]]:
        if not block or not isinstance(block, str):
            return None
        try:
            import re as _re
            out = {}
            for k in ("classic_refs", "kyukokumei_refs", "food_refs"):
                m = _re.search(rf"{k}\s*=\s*(\d+)", block)
                if m:
                    out[k] = int(m.group(1))
            return out or None
        except Exception:
            return None
    
    def _update_history(self, speaker: str, content: str) -> None:
        """履歴を更新"""
        self.state.history.append({
            "role": speaker,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "turn": self.state.turn_count
        })
    
    def _update_metrics(self, turn_time: float, response_length: int) -> None:
        """メトリクスを更新"""
        self.state.metrics["last_turn_time"] = turn_time
        self.state.metrics["total_time"] = time.time() - self.state.start_time
        self.state.total_tokens += response_length  # 簡易的なトークン数
        
        # 平均応答時間を計算
        if "turn_times" not in self.state.metrics:
            self.state.metrics["turn_times"] = []
        self.state.metrics["turn_times"].append(turn_time)
        self.state.metrics["avg_response_time"] = sum(self.state.metrics["turn_times"]) / len(self.state.metrics["turn_times"])
    
    def pause(self) -> None:
        """対話を一時停止"""
        self.state.is_paused = True
    
    def resume(self) -> None:
        """対話を再開"""
        self.state.is_paused = False
    
    def stop(self) -> None:
        """対話を停止"""
        self.state.is_active = False
    
    def reset(self) -> None:
        """状態をリセット"""
        self.state = DialogueState()
        # エージェントのメモリをクリア（メソッドが存在する場合）
        for agent in self.agents.values():
            if hasattr(agent, 'reset'):
                agent.reset()
        # Directorのセッションキャッシュをクリア
        try:
            if self.director and hasattr(self.director, 'entity_cache'):
                self.director.entity_cache.clear()
            if self.director and hasattr(self.director, 'pending_soft_ack'):
                self.director.pending_soft_ack.clear()
        except Exception:
            pass
    
    def get_state_summary(self) -> Dict[str, Any]:
        """状態サマリを取得"""
        if not self.state:
            return {}
            
        return {
            "turn_count": self.state.turn_count,
            "phase": self.state.phase,
            "total_tokens": self.state.total_tokens,
            "avg_response_time": self.state.metrics.get("avg_response_time", 0),
            "is_active": self.state.is_active,
            "is_paused": self.state.is_paused,
            "current_phase": self.state.phase,
            "phase_progress": min(self.state.turn_count / self.config.max_turns, 1.0) if self.config else 0
        }
    
    def get_history(self) -> List[Dict]:
        """対話履歴を取得"""
        return self.state.history if self.state else []
    
    def export_session(self, filepath: Optional[str] = None) -> str:
        """セッションをエクスポート"""
        if not self.state or not self.config:
            return json.dumps({"error": "No active session"})
            
        export_data = {
            "config": {
                "theme": self.config.theme,
                "agents": [self.config.agent1_name, self.config.agent2_name],
                "max_turns": self.config.max_turns
            },
            "state": {
                "turn_count": self.state.turn_count,
                "phase": self.state.phase,
                "total_tokens": self.state.total_tokens,
                "duration": self.state.metrics.get("total_time", 0)
            },
            "history": self.state.history,
            "metrics": self.state.metrics
        }
        
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            return filepath
        else:
            return json.dumps(export_data, ensure_ascii=False, indent=2)