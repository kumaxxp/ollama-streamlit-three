"""
Agent - 対話エージェントの実装
キャラクター設定に基づく自然な対話生成
"""

import json
import logging
import os
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class Agent:
    """
    対話エージェント
    一般的なキャラクター（女子高生、会社員など）として振る舞う
    """
    
    def __init__(
        self,
        agent_id: str,
        character_type: str,
        model_name: str = "qwen2.5:7b",
        temperature: float = 0.7,
        ollama_client = None
    ):
        self.agent_id = agent_id
        self.character_type = character_type
        self.model_name = model_name
        self.temperature = temperature
        self.client = ollama_client
        
        # キャラクター設定を読み込み
        self.character = self._load_character(character_type)
        
        # セッション情報
        self.session_context = {}
        self.turn_directives = []
        self.memory = []  # 発言履歴
        
        logger.info(f"Agent {agent_id} initialized as {self.character['name']}")
    
    def _load_character(self, character_type: str) -> Dict:
        """キャラクター設定を読み込み"""
        import os
        
        # 複数のパスを試す
        possible_paths = [
            'config/characters.json',
            './config/characters.json',
            '../config/characters.json',
            '../../config/characters.json',
            os.path.join(os.path.dirname(__file__), '../../config/characters.json')
        ]
        
        characters_data = None
        used_path = None
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        characters_data = json.load(f)
                        used_path = path
                        logger.info(f"Characters loaded from: {path}")
                        break
                except Exception as e:
                    logger.error(f"Error loading from {path}: {e}")
                    continue
        
        if not characters_data:
            logger.error(f"characters.json not found in any of these paths: {possible_paths}")
            logger.error(f"Current working directory: {os.getcwd()}")
            return self._get_default_character()
        
        # キャラクターを取得
        if 'characters' in characters_data:
            if character_type in characters_data['characters']:
                logger.info(f"Character '{character_type}' loaded successfully")
                character = characters_data['characters'][character_type]

                # 外部ファイル参照があれば読み込んで prompt_template にセットする
                if 'prompt_template_file' in character:
                    # 可能な相対パス候補: characters.json のディレクトリ基準、およびワークディレクトリ基準
                    base_dirs = []
                    if used_path:
                        base_dirs.append(os.path.dirname(os.path.abspath(used_path)))
                    base_dirs.extend([os.getcwd(), os.path.abspath('.')])

                    file_path = character.get('prompt_template_file')
                    found = False
                    for base in base_dirs:
                        candidate = os.path.join(base, file_path)
                        if os.path.exists(candidate):
                            try:
                                with open(candidate, 'r', encoding='utf-8') as pf:
                                    character['prompt_template'] = pf.read()
                                logger.info(f"Loaded prompt template from file: {candidate}")
                                found = True
                                break
                            except Exception as e:
                                logger.error(f"Failed to read prompt template file {candidate}: {e}")
                                continue
                    if not found:
                        # そのままファイルパスを試す
                        try:
                            if os.path.exists(file_path):
                                with open(file_path, 'r', encoding='utf-8') as pf:
                                    character['prompt_template'] = pf.read()
                                logger.info(f"Loaded prompt template from file: {file_path}")
                                found = True
                        except Exception as e:
                            logger.error(f"Failed to read prompt template file {file_path}: {e}")

                return character
            else:
                logger.warning(f"Character type '{character_type}' not found in characters.json")
                logger.warning(f"Available characters: {list(characters_data['characters'].keys())}")
                return self._get_default_character()
        else:
            logger.error("'characters' key not found in JSON")
            return self._get_default_character()
    
    def _get_default_character(self) -> Dict:
        """デフォルトキャラクター（フォールバック用）"""
        return {
            "name": "対話参加者",
            "personality": "中立的で建設的な議論を心がける",
            "speaking_style": "丁寧で分かりやすい",
            "background": "一般的な知識を持つ対話者",
            "values": "相互理解と建設的な議論",
            "behavioral_patterns": {
                "greeting": ["こんにちは", "よろしくお願いします"],
                "agreement": ["なるほど", "その通りですね"],
                "disagreement": ["別の見方もあるかもしれません", "私はこう思います"]
            }
        }
    
    def set_session_context(self, theme: str, goal: str, phase: str):
        """セッションコンテキストを設定"""
        self.session_context = {
            "theme": theme,
            "goal": goal,
            "phase": phase,
            "start_time": datetime.now().isoformat()
        }
    
    def add_directive(self, instruction: str, attention_points: List[str]):
        """Directorからの指示を追加"""
        self.turn_directives.append({
            "instruction": instruction,
            "attention_points": attention_points,
            "timestamp": datetime.now().isoformat()
        })
    
    def _build_system_prompt(self) -> str:
        """システムプロンプトを構築（UIで表示するためprintは行わない）"""
        # キャラクターのプロンプトテンプレートを使用
        if 'prompt_template' in self.character:
            template = self.character['prompt_template']
            system_prompt = template.format(
                name=self.character['name'],
                personality=self.character.get('personality', ''),
                speaking_style=self.character.get('speaking_style', ''),
                background=self.character.get('background', ''),
                values=self.character.get('values', '')
            )
            return self._append_required_constraints(system_prompt)
        # テンプレートがない場合は標準形式
        system_prompt = (
            f"あなたは{self.character['name']}です。\n"
            f"\n"
            f"【基本設定】\n"
            f"性格: {self.character.get('personality', '未設定')}\n"
            f"話し方: {self.character.get('speaking_style', '自然な日本語')}\n"
            f"背景: {self.character.get('background', '一般的な背景')}\n"
            f"価値観: {self.character.get('values', '建設的な対話')}\n"
            f"\n"
            f"【重要な指示】\n"
            f"- あなたのキャラクターを一貫して保ってください\n"
            f"- 相手の発言をよく聞き、それに応答してください\n"
            f"- 自然な対話を心がけてください\n"
            f"- 議論を建設的に進めてください\n"
        )
        return self._append_required_constraints(system_prompt)

    # （不要な日本語生文字列を削除済み）
    
    def build_prompt(self, context: Dict) -> str:
        """
        対話用プロンプトを構築
        
        Args:
            context: {
                "opponent_name": str,
                "opponent_message": str,
                "recent_history": List[Dict],
                "director_instruction": Optional[str]
            }
        """
        opponent_name = context.get('opponent_name', '相手')
        opponent_message = context.get('opponent_message', '')
        recent_history = context.get('recent_history', [])
        director_instruction = context.get('director_instruction', '')
        metrics_block = context.get('metrics_block')
        director_findings = context.get('director_findings')
        
        # 基本プロンプト（ユーザー表示用。指定の体裁に合わせる）
        prompt_parts = [
            f"あなたは{self.character['name']}として、{opponent_name}と対話しています。",
            f"\nテーマ: {self.session_context.get('theme', '自由討論')}",
            f"現在のフェーズ: {self.session_context.get('phase', 'exploration')}\n"
        ]
        
        # 最近の履歴があれば追加
        if recent_history:
            prompt_parts.append("\n【これまでの対話】")
            for entry in recent_history[-3:]:  # 最新3つ
                speaker = entry.get('speaker', '不明')
                message = entry.get('message', '')
                prompt_parts.append(f"{speaker}: 「{message}」")
        
        # 相手の最新メッセージ
        if opponent_message:
            prompt_parts.append(f"\n【{opponent_name}の発言】")
            prompt_parts.append(f"「{opponent_message}」")
        
        # Director指示があれば追加
        if director_instruction:
            prompt_parts.append(f"\n【アドバイス】")
            prompt_parts.append(director_instruction)

        # メトリクス（User Prompt専用表示）
        if isinstance(metrics_block, str) and metrics_block.strip():
            prompt_parts.append("\n【モニタリング】")
            prompt_parts.append(metrics_block)

    # Directorの検出/検証/レビュー補足があれば追加（1ターンのみ有効）
        if isinstance(director_findings, dict):
            # エンティティ検証
            if director_findings.get('entity_name'):
                en = director_findings.get('entity_name')
                vd = director_findings.get('verdict') or 'AMBIGUOUS'
                ev = director_findings.get('evidence')
                ex = director_findings.get('evidence_excerpt')
                prompt_parts.append("\n【検証補足（Director）】")
                prompt_parts.append(f"対象: 『{en}』 / 判定: {vd}")
                if ev:
                    prompt_parts.append(f"根拠URL: {ev}")
                if ex:
                    prompt_parts.append("根拠要約: " + ex)
                prompt_parts.append("必要に応じてこの情報をやわらかく活用してください（確認・補足など）。")
            # ホリスティックレビュー（文章）
            if director_findings.get('holistic_text'):
                prompt_parts.append("\n【レビュー補足（Director）】")
                prompt_parts.append(str(director_findings.get('holistic_text')))
                # 指定のガイダンス文を追加
                prompt_parts.append("ここにある指摘は必ずツッコミ材料として短く活用してください。")
                prompt_parts.append("解説や説明にはせず、一言の指摘や皮肉に変換してください。")
            # 作品検証/参考のガイダンス（先頭に一度だけ）
            if director_findings.get('works_detected') or director_findings.get('wiki_snippets'):
                prompt_parts.append("\n【作品検証（Director）】【参考（Wikipedia検索）}")
                prompt_parts.append("ここにある情報は事実確認用。必要に応じて一言触れてからツッコむ程度に使ってください。")
            # 作品検証があれば提示
            if director_findings.get('works_detected'):
                ws = director_findings.get('works_detected')
                try:
                    if isinstance(ws, list) and ws:
                        prompt_parts.append("\n【作品検証（Director）】")
                        for item in ws[:2]:
                            if isinstance(item, dict):
                                t = item.get('title')
                                v = item.get('verdict')
                                u = item.get('url')
                                if t and v:
                                    line = f"『{t}』: 判定 {v}"
                                    if u:
                                        line += f" / URL: {u}"
                                    prompt_parts.append(line)
                        prompt_parts.append("存在が曖昧/NGのものは、誤記や架空の可能性を短く確認してから話を進めてください。")
                except Exception:
                    pass
            # Wikipediaスニペット（必要ならURLを1つだけ）
            if director_findings.get('wiki_snippets'):
                sn = director_findings.get('wiki_snippets')
                try:
                    if isinstance(sn, list) and sn:
                        prompt_parts.append("\n【参考（Wikipedia検索）}")
                        s0 = sn[0]
                        if isinstance(s0, dict):
                            q = s0.get('query')
                            t = s0.get('title')
                            u = s0.get('url')
                            ex = s0.get('excerpt')
                            if q:
                                prompt_parts.append(f"検索: {q}")
                            if t:
                                prompt_parts.append(f"候補: {t}")
                            if ex:
                                prompt_parts.append("要約: " + str(ex)[:200])
                            if u:
                                prompt_parts.append(f"URL: {u}")
                except Exception:
                    pass
            # Review directives の短い実行ブロック
            if director_findings.get('review_block'):
                prompt_parts.append("\n【From Director Review】")
                prompt_parts.append(str(director_findings.get('review_block')))
        
        # 最新のDirective（長さ指示を含む）
        if self.turn_directives:
            latest_directive = self.turn_directives[-1]
            prompt_parts.append(f"\n【今回の注意点】")
            prompt_parts.append(latest_directive.get('instruction', ''))
            points = latest_directive.get('attention_points') or []
            if isinstance(points, list) and points:
                prompt_parts.append("- " + "\n- ".join(points))
            
            # 長さ指示が含まれているかチェック
            length_guidance = None
            for point in points:
                if "簡潔" in point:
                    length_guidance = "50-100文字程度で簡潔に"
                elif "詳細" in point:
                    length_guidance = "200-300文字程度で詳しく"
                elif "標準" in point:
                    length_guidance = "100-200文字程度で"
            
            if length_guidance:
                prompt_parts.append(f"\n【応答の長さ】{length_guidance}")

        # 応答/開始指示（初回は開始、以降は応答）
        is_opening = (not recent_history) and (not opponent_message)
        if is_opening:
            prompt_parts.append(f"\n上記を踏まえて、{self.character['name']}として会話を開始してください。")
            prompt_parts.append("テーマに即した導入を述べてください。挨拶は不要です。2〜4文、200-300文字を目安に。短い問いかけ指示があっても最初は導入重視で長めに。")
        else:
            prompt_parts.append(f"\n上記を踏まえて、{opponent_name}に対して{self.character['name']}として応答してください。")
            prompt_parts.append("自然で、あなたらしい発言を心がけてください。")

        # フォーマット指示を追加（UI表示用、生成時は構造化メッセージを使用）
        prompt_parts.append("\n【応答形式の注意】")
        prompt_parts.append("- 挨拶不要。すぐに本題。")
        prompt_parts.append("- 箇条書きや見出しは使わない。")
        prompt_parts.append("- マークダウン記号を使わない。")
        prompt_parts.append("- 相手のセリフや名前は書かない。")
        prompt_parts.append("- キャラクタ名や役割（やな:、あゆ: 等）を出力に含めない。")
        prompt_parts.append("- 出力はあなた自身の発言のみ。")
        # 禁止事項
        prompt_parts.append("\n【禁止事項】")
        ban_quotes = False
        try:
            if isinstance(director_findings, dict):
                rd = director_findings.get('review_directives') or {}
                if 'ban_classic_quotes' in (rd.get('required_actions') or []):
                    ban_quotes = True
        except Exception:
            ban_quotes = False
        if ban_quotes:
            prompt_parts.append("古典/名言/物語の引用は禁止（『〜曰く』『〜のように』等）。自分の言葉で述べる。")
        prompt_parts.append("褒め言葉は禁止")

        return "\n".join(prompt_parts)
    
    async def generate_response(self, context: Dict, system_prompt: Optional[str] = None, user_prompt: Optional[str] = None) -> str:
        """
        応答を生成
        
        Args:
            context: 対話コンテキスト
        
        Returns:
            生成された応答
        """
        # system prompt（必須制約を付与）
        if system_prompt is None:
            system_prompt = self._build_system_prompt()
        else:
            system_prompt = self._append_required_constraints(system_prompt)

        # 構造化メッセージに変換（台本風混入防止）
        messages = self._build_structured_messages(context, system_prompt)
        # 直近の指示（ディレクター変換済みテキスト）を追加（別system）
        if self.turn_directives:
            latest = self.turn_directives[-1]
            directive_text = latest.get("instruction", "")
            if directive_text:
                messages.insert(1, {"role": "system", "content": directive_text})
        
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "top_p": 0.95,
                    "seed": None
                },
                stream=False
            )
            generated_text = response['message']['content']

            # Post-check（キャラ混入検査）
            own_name = self.character['name']
            other_names: List[str] = []
            try:
                # opponent 名（コンテキストから）
                if context.get('opponent_name'):
                    other_names.append(context['opponent_name'])
            except Exception:
                pass

            # 検知/修正のために短縮名エイリアスも対象に含める
            expanded_others: List[str] = []
            for n in other_names:
                expanded_others.extend(self._expand_name_aliases(n))
            other_names = list(dict.fromkeys([*other_names, *expanded_others]))

            if self._check_character_leak(generated_text, own_name, other_names):
                # 1回だけ再生成（違反を明記）
                retry_messages = messages + [{
                    "role": "system",
                    "content": "直前の出力に相手の発言や名前が含まれていました。自分の発言のみ（1〜2文）で、相手のセリフや名前を書かずに再出力してください。"
                }]
                try:
                    response = self.client.chat(
                        model=self.model_name,
                        messages=retry_messages,
                        options={
                            "temperature": self.temperature,
                            "top_p": 0.95,
                            "seed": None
                        },
                        stream=False
                    )
                    retried = response['message']['content']
                    if self._check_character_leak(retried, own_name, other_names):
                        # 強制カット
                        generated_text = self._force_cut_single_speaker(retried, own_name, other_names)
                    else:
                        generated_text = retried
                except Exception:
                    generated_text = self._force_cut_single_speaker(generated_text, own_name, other_names)
            
            # メモリに追加
            self.memory.append({
                "role": "assistant",
                "content": generated_text,
                "timestamp": datetime.now().isoformat()
            })
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Response generation error for {self.agent_id}: {e}")
            # Return structured error info so controller/UI can surface it
            return {
                "error": True,
                "message": self._get_fallback_response(context),
                "detail": str(e)
            }

    # ====== キャラ混在対策ユーティリティ ======
    def _append_required_constraints(self, system_prompt: str) -> str:
        """system prompt末尾に必須制約文言を付与"""
        name = self.character.get('name', 'あなた')
        req = (
            f"\n【厳守事項】\n"
            f"- あなたは『{name}』です。\n"
            f"- 他キャラクタの発言や台本形式の会話文は書かないでください。\n"
            f"- 出力はあなた自身の発言のみ。相手のセリフや名前は書かない。\n"
            f"- キャラクタ名や役割（{name}やA/B等）を出力に含めない。\n"
            f"- 出力の先頭に『名前:』や『A:』『B:』などのラベルを付けない。\n"
        )
        if req.strip() in system_prompt:
            return system_prompt
        return f"{system_prompt}\n{req}"

    def _build_structured_messages(self, context: Dict, system_prompt: str) -> List[Dict[str, str]]:
        """仕様 2.3 に沿ったメッセージ配列を構築"""
        own_name = self.character.get('name', '自分')
        opponent_name = context.get('opponent_name', '相手')
        opponent_msg = context.get('opponent_message', '') or ''
        recent = context.get('recent_history', []) or []

        # 直近の自分の発話を取得
        own_last: Optional[str] = None
        for item in reversed(recent):
            if item.get('speaker') == own_name:
                own_last = str(item.get('message', ''))
                break

        messages: List[Dict[str, str]] = []
        messages.append({"role": "system", "content": system_prompt})

        # セッションコンテキスト（テーマ/目的/フェーズ）を簡潔に明示
        theme = (self.session_context or {}).get('theme') or context.get('theme')
        goal = (self.session_context or {}).get('goal') or context.get('goal')
        phase = (self.session_context or {}).get('phase') or context.get('phase')
        ctx_lines: List[str] = []
        if theme:
            ctx_lines.append(f"- テーマ: {theme}")
        if goal:
            ctx_lines.append(f"- 目的: {goal}")
        if phase:
            ctx_lines.append(f"- 現在フェーズ: {phase}")
        if ctx_lines:
            messages.append({
                "role": "system",
                "content": "【セッションコンテキスト】\n" + "\n".join(ctx_lines)
            })

        # Directorの検出/検証情報があれば、補足として伝える（次ターンのみ）
        findings = context.get("director_findings")
        if isinstance(findings, dict) and findings.get("entity_name"):
            fn = findings.get("entity_name")
            vt = findings.get("verdict")
            ev = findings.get("evidence")
            ex = findings.get("evidence_excerpt")
            note_lines = [
                "【検証補足（Director）】",
                f"直前の発話に登場した『{fn}』について、検証結果: {vt or 'AMBIGUOUS'}",
            ]
            if ev:
                note_lines.append(f"根拠URL: {ev}")
            if ex:
                note_lines.append("根拠要約: " + ex)
            note_lines.append("必要に応じてこの情報を活用してください。（やわらかい確認/補足が望ましい）")
            messages.append({"role": "system", "content": "\n".join(note_lines)})

        # 追加: 研究結果(research)の共有（URLと短い抜粋）
        if isinstance(findings, dict) and findings.get("research"):
            r = findings["research"]
            lines = ["【補足検索（Director）】"]
            if r.get("query"):
                lines.append(f"クエリ: {r.get('query')}")
            if r.get("verdict"):
                lines.append(f"判定: {r.get('verdict')}")
            if r.get("evidence"):
                lines.append(f"参考URL: {r.get('evidence')}")
            if r.get("evidence_excerpt"):
                lines.append("要約: " + str(r.get("evidence_excerpt")))
            lines.append("必要なら事実関係を短く確認してください。")
            messages.append({"role": "system", "content": "\n".join(lines)})

        # review_directives が来ていたら、強制的な system 指示を前段で注入（例: 引用禁止）
        try:
            if isinstance(findings, dict) and findings.get('review_directives'):
                rd = findings.get('review_directives') or {}
                reqs = rd.get('required_actions') or []
                avoids = rd.get('avoid') or []
                lines = ["【遵守指示（Director）】"]
                if 'ban_classic_quotes' in reqs or 'overuse_classics' in avoids:
                    lines.append("- 古典/名言/物語の引用は禁止。『〜曰く』『〜のように』等を避け、自分の言葉で要点だけ述べる。")
                if 'one_concise_question' in (rd.get('ensure') or []):
                    lines.append("- 応答の最後に短い質問を1つだけ添える。")
                if len(lines) > 1:
                    messages.append({"role": "system", "content": "\n".join(lines)})
        except Exception:
            pass

        # 開始/通常の分岐
        is_opening = (len(recent) == 0 and not opponent_msg)
        if is_opening:
            # 開始専用の追加system（短く本題へ・名前や台本禁止を再強調）
            start_lines = [
                "【開始指示】",
                "これは会話の最初の発話です。",
                "- 挨拶は不要です。すぐに本題に入ってください。",
                "- テーマに即した導入を述べてください。",
                "- 相手のセリフや名前、台本形式（A: / B: など）は書かないでください。",
                "- 出力の先頭に名前やラベル（やな:、あゆ:、A:、B:など）を付けないでください。",
            ]
            messages.append({"role": "system", "content": "\n".join(start_lines)})
            # userには中立のトリガーだけを与える（相手名は入れない）
            trigger = "開始: テーマに基づく導入を述べてください。"
            messages.append({"role": "user", "content": trigger})
        else:
            # 直近の自分の発話（あれば assistant）
            if own_last:
                messages.append({"role": "assistant", "content": own_last})
            # 相手の最新発話（user）
            if opponent_msg:
                messages.append({"role": "user", "content": opponent_msg})
            else:
                # 最低限の起点（従来どおり。ただし冒頭ではis_opening側が使われる）
                messages.append({"role": "user", "content": "（前の続き）"})

        return messages

    def _check_character_leak(self, output_text: str, own_name: str, other_names: List[str]) -> bool:
        import re
        text = output_text.strip()
        # 他キャラ名（エイリアス含む）がラベル様式で混入
        names_other = []
        for n in other_names:
            names_other += self._expand_name_aliases(n)
        names_other = [x for x in dict.fromkeys(names_other) if x]

        # ラベル混入（先頭/空白後）
        if re.search(r"(^|\s)[AB][：:]", text):
            return True
        name_alt = []
        name_alt += [re.escape(x) for x in self._expand_name_aliases(own_name)]
        name_alt += [re.escape(n) for n in names_other]
        if name_alt:
            pattern = rf"(^|\s)({'|'.join(name_alt)})[：:]"
            if re.search(pattern, text):
                return True
        # 他キャラ名が文中に明確に現れるケース（短縮名含む）
        for n in names_other:
            if n and (n + ":") in text or (n + "：") in text:
                return True
        return False

    def _force_cut_single_speaker(self, text: str, own_name: str, other_names: List[str]) -> str:
        """緊急カット: 他名やラベルを除去し、自分の発話だけに整形"""
        import re
        s = text
        # 行頭や空白後の「名前:」を除去（自分/他人どちらも）
        names = []
        names += self._expand_name_aliases(own_name)
        for n in other_names:
            names += self._expand_name_aliases(n)
        if names:
            pattern = rf"(^|\s)({'|'.join([re.escape(n) for n in names])})[：:]\s*"
            s = re.sub(pattern, " ", s).strip()
        # 相手名を含む単純な書き起こしの除去
        for n in names:
            if not n:
                continue
            s = s.replace(f"{n}：", "").replace(f"{n}:", "")
        # 2文に制限
        parts = re.split(r"[。.!?？！]+", s)
        parts = [p for p in parts if p.strip()]
        return ("。".join(parts[:2]) + ("。" if parts[:2] else "")).strip()

    def _expand_name_aliases(self, name: Optional[str]) -> List[str]:
        """表示名から括弧付き注釈などを除いた短縮名エイリアスを返す"""
        if not name:
            return []
        try:
            n = str(name)
            base = n.split("（")[0].split("(")[0].strip()
            # 記号や空白の正規化
            alts = [n.strip()]
            if base and base not in alts:
                alts.append(base)
            # 全角コロン・半角コロンを伴うラベルマッチに備えて末尾空白も除く
            return [a for a in alts if a]
        except Exception:
            return [str(name)] if name else []
    
    def _get_fallback_response(self, context: Dict) -> str:
        """エラー時のフォールバック応答"""
        opponent_name = context.get('opponent_name', '相手')
        
        # キャラクターに応じた基本応答
        if 'high_school' in self.character_type:
            return f"{opponent_name}さん、それってすごく難しいですね...でも面白い！"
        elif 'office_worker' in self.character_type:
            return f"{opponent_name}さんの意見、実務的な観点から興味深いですね。"
        elif 'college_student' in self.character_type:
            return f"{opponent_name}さん、その視点は新鮮ですね。もう少し考えてみます。"
        else:
            return f"{opponent_name}さん、なるほど、そういう見方もありますね。"
    
    def get_character_info(self) -> Dict:
        """キャラクター情報を取得"""
        return {
            "id": self.agent_id,
            "name": self.character['name'],
            "type": self.character_type,
            "personality": self.character.get('personality', ''),
            "background": self.character.get('background', '')
        }
    
    def reset(self):
        """エージェントをリセット"""
        self.session_context = {}
        self.turn_directives = []
        self.memory = []
        logger.info(f"Agent {self.agent_id} reset")