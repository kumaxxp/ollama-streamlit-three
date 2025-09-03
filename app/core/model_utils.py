"""
Model Selection Utilities
Ollamaモデルの動的取得と管理（安全化版）
"""

import json
import logging
import subprocess
from typing import List, Dict, Optional, Tuple
# ollama Python クライアントが無い環境でも落ちないように安全化
try:
    import ollama  # type: ignore
    _HAS_OLLAMA = True
except Exception:
    ollama = None  # type: ignore
    _HAS_OLLAMA = False

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Ollamaモデルの管理クラス
    """
    
    def __init__(self):
        # クライアントは存在する場合のみ初期化
        self.client = None
        try:
            if _HAS_OLLAMA:
                self.client = ollama.Client()  # type: ignore
        except Exception:
            self.client = None
        self.model_config = self._load_model_config()
        
    def _load_model_config(self) -> Dict:
        """モデル設定ファイルを読み込み"""
        try:
            with open('config/model_config.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("model_config.json not found, using defaults")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading model config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """デフォルト設定"""
        return {
            "recommended_models": {
                "dialogue_agents": {
                    "japanese_optimized": [
                        {"name": "qwen2.5:7b", "description": "日本語推奨"},
                        {"name": "qwen2.5:14b", "description": "高品質"}
                    ]
                }
            }
        }
    
    def get_available_models(self) -> List[str]:
        """
        Ollamaで利用可能なモデル一覧を取得
        
        Returns:
            利用可能なモデル名のリスト
        """
        try:
            # Ollama Python Clientを使用（可能なら）
            if self.client is not None:
                response = self.client.list()
            else:
                raise RuntimeError("ollama client unavailable")
            
            models = []
            if 'models' in response:
                for model in response['models']:
                    # モデル名を取得（タグ付き）
                    model_name = model.get('name', '')
                    if model_name:
                        models.append(model_name)
            
            # 重複を削除してソート
            models = sorted(list(set(models)))
            
            logger.info(f"Found {len(models)} models in Ollama")
            return models
            
        except Exception as e:
            logger.error(f"Failed to get models via client: {e}")
            # フォールバック: CLIコマンドを試す
            return self._get_models_via_cli()
    
    def _get_models_via_cli(self) -> List[str]:
        """
        CLIコマンドでモデル一覧を取得（フォールバック）
        """
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                models = []
                
                # ヘッダー行をスキップ
                for line in lines[1:]:
                    if line.strip():
                        # 最初の空白までをモデル名として取得
                        parts = line.split()
                        if parts:
                            model_name = parts[0]
                            models.append(model_name)
                
                return sorted(models)
            
        except Exception as e:
            logger.error(f"Failed to get models via CLI: {e}")
        
        # 最終フォールバック
        return self._get_fallback_models()
    
    def _get_fallback_models(self) -> List[str]:
        """
        フォールバック用のモデルリスト（configベース）
        """
        # configから候補を収集し、順序をできるだけ保って重複排除
        candidates: List[str] = []

        # defaultのfallbackがあれば最上位
        try:
            default_models = self.model_config.get("default_models", {})
            fb = default_models.get("fallback")
            if fb:
                candidates.append(fb)
            # agent/directorのデフォルトも候補に
            for k in ("agent", "director"):
                mv = default_models.get(k)
                if mv:
                    candidates.append(mv)
        except Exception:
            pass

        # model_selection_rulesのpreferred_modelsを追加
        try:
            rules = self.model_config.get("model_selection_rules", {})
            for key in ("dialogue_agent", "director"):
                pref = rules.get(key, {}).get("preferred_models", [])
                candidates.extend(pref)
        except Exception:
            pass

        # production_modelsの一覧をpriority順（numberが小さいほど高優先）で追加
        try:
            prod = self.model_config.get("production_models", {})
            all_items = []
            for group in prod.values():
                if isinstance(group, list):
                    all_items.extend(group)
            # priorityのないものは後ろ
            all_items.sort(key=lambda x: (x.get("priority") is None, x.get("priority", 1_000_000)))
            candidates.extend([item.get("name") for item in all_items if item.get("name")])
        except Exception:
            pass

        # 重複排除（順序保持）
        seen = set()
        ordered: List[str] = []
        for c in candidates:
            if c and c not in seen:
                seen.add(c)
                ordered.append(c)

        # 最低限のフォールバック
        if not ordered:
            ordered = ["qwen:7b", "gemma2:2b", "llama3.2:3b"]

        return ordered

    def get_fallback_models(self) -> List[str]:
        """UI等から利用する公開用フォールバック候補取得"""
        return self._get_fallback_models()
    
    def get_sorted_models(self, available_models: List[str]) -> List[str]:
        """
        モデルを推奨順にソート
        
        Args:
            available_models: 利用可能なモデルリスト
            
        Returns:
            ソート済みモデルリスト
        """
        # configから優先順リストを構築
        priority_order: List[str] = []

        # 1) production_modelsのpriority順
        try:
            prod = self.model_config.get("production_models", {})
            all_items = []
            for group in prod.values():
                if isinstance(group, list):
                    all_items.extend(group)
            if all_items:
                all_items.sort(key=lambda x: (x.get("priority") is None, x.get("priority", 1_000_000)))
                priority_order = [item.get("name") for item in all_items if item.get("name")]
        except Exception:
            pass

        # 2) 次にmodel_selection_rulesのpreferred_models
        if not priority_order:
            try:
                rules = self.model_config.get("model_selection_rules", {})
                priority_order = []
                for key in ("dialogue_agent", "director"):
                    priority_order.extend(rules.get(key, {}).get("preferred_models", []))
            except Exception:
                priority_order = []

        # 3) まだ空ならフォールバック候補
        if not priority_order:
            priority_order = self._get_fallback_models()

        # 優先モデルと利用可能モデルの交差
        priority_available = [m for m in priority_order if m in available_models]

        # その他のモデル
        other_models = sorted([m for m in available_models if m not in set(priority_available)])

        return priority_available + other_models
    
    def get_model_info(self, model_name: str) -> Dict:
        """
        モデルの詳細情報を取得
        
        Args:
            model_name: モデル名
            
        Returns:
            モデル情報の辞書
        """
        try:
            if self.client is None:
                raise RuntimeError("ollama client unavailable")
            info = self.client.show(model_name)
            
            return {
                "name": model_name,
                "size": info.get("details", {}).get("parameter_size", "不明"),
                "quantization": info.get("details", {}).get("quantization_level", "不明"),
                "family": info.get("details", {}).get("family", "不明"),
                "format": info.get("details", {}).get("format", "不明")
            }
            
        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {e}")

            # 可能なら CLI 経由で最低限の情報を取得
            try:
                import subprocess
                result = subprocess.run(["ollama", "show", model_name], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    txt = result.stdout.strip()
                    return {"name": model_name, "raw": txt[:2000]}
            except Exception:
                pass

            # 設定ファイルから情報を取得
            for category in self.model_config.get("recommended_models", {}).get("dialogue_agents", {}).values():
                for model in category:
                    if model.get("name") == model_name:
                        return {
                            "name": model_name,
                            "description": model.get("description", ""),
                            "vram_required": model.get("vram_required", "不明")
                        }
            
            return {"name": model_name, "info": "詳細情報なし"}
    
    def get_recommended_temperature(self, model_name: str, use_case: str = "dialogue") -> float:
        """
        モデルと用途に応じた推奨温度を取得（configを単一ソースとして参照）
        優先度: 1) production_models 個別設定 > 2) model_selection_rules 用途別 > 3) 既定値

        Args:
            model_name: モデル名（タグ含む）
            use_case: 用途（"agent"|"dialogue"|"creative"|"director"）

        Returns:
            推奨温度（0.0〜1.0想定）
        """
        try:
            # 正規化: 用途をagent/dialogue/creative/directorに丸める
            uc = (use_case or "dialogue").lower()
            if uc in ("agent", "dialogue"):
                uc_key = "agent"
            elif uc == "creative":
                uc_key = "agent"  # creativeはagent系の一種として扱い、用途別既定でカバー
            else:
                uc_key = "director"

            # 1) production_modelsから個別温度
            prod = self.model_config.get("production_models", {})
            for group in prod.values():
                if isinstance(group, list):
                    for item in group:
                        if item.get("name") == model_name:
                            temp = item.get("temperature", {})
                            if uc_key == "director":
                                val = temp.get("director")
                            else:
                                val = temp.get("agent")
                            if isinstance(val, (int, float)):
                                return float(val)

            # 2) 用途別ルールの推奨
            rules = self.model_config.get("model_selection_rules", {})
            if uc_key == "director":
                val = rules.get("director", {}).get("recommended_temperature")
            else:
                # agent/dialogue/creative は dialogue_agent に丸める
                val = rules.get("dialogue_agent", {}).get("recommended_temperature")
            if isinstance(val, (int, float)):
                return float(val)

        except Exception:
            # 何があっても既定にフォールバック
            pass

        # 3) 既定値（後方互換）
        if uc_key == "director":
            return 0.3
        elif use_case == "creative":
            return 0.7
        else:
            return 0.6
    
    def check_model_exists(self, model_name: str) -> bool:
        """
        モデルが存在するかチェック
        
        Args:
            model_name: モデル名
            
        Returns:
            存在する場合True
        """
        available = self.get_available_models()
        return model_name in available
    
    def pull_model(self, model_name: str) -> bool:
        """
        モデルをダウンロード
        
        Args:
            model_name: モデル名
            
        Returns:
            成功した場合True
        """
        try:
            logger.info(f"Pulling model: {model_name}")
            if self.client is not None:
                self.client.pull(model_name)
                return True
            # フォールバック: CLI
            import subprocess
            result = subprocess.run(["ollama", "pull", model_name], capture_output=True, text=True)
            return result.returncode == 0

        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False
    
    def get_model_recommendations(self, vram_gb: int = 8) -> List[Dict]:
        """
        VRAM容量に基づくモデル推奨
        
        Args:
            vram_gb: VRAM容量（GB）
            
        Returns:
            推奨モデルのリスト
        """
        recommendations = []
        
        for category in self.model_config.get("recommended_models", {}).get("dialogue_agents", {}).values():
            for model in category:
                vram_required = model.get("vram_required", "0GB")
                required_gb = int(vram_required.replace("GB", ""))
                
                if required_gb <= vram_gb:
                    recommendations.append({
                        "name": model.get("name"),
                        "description": model.get("description"),
                        "vram_required": vram_required,
                        "recommended": model.get("recommended", False)
                    })
        
        # 推奨順にソート
        recommendations.sort(key=lambda x: (not x.get("recommended"), x.get("name")))
        
        return recommendations