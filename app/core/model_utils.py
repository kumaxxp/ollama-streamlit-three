"""
Model Selection Utilities
Ollamaモデルの動的取得と管理
"""

import json
import logging
import subprocess
from typing import List, Dict, Optional, Tuple
import ollama

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Ollamaモデルの管理クラス
    """
    
    def __init__(self):
        self.client = ollama.Client()
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
            # Ollama Python Clientを使用
            response = self.client.list()
            
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
        フォールバック用のモデルリスト
        """
        return [
            # Qwen系列
            "qwen2.5:7b",
            "qwen2.5:14b",
            "qwen2.5:32b",
            "qwen2.5:72b",
            
            # Gemma系列
            "gemma2:2b",
            "gemma2:9b", 
            "gemma2:27b",
            "gemma3:3b",
            "gemma3:9b",
            "gemma3:12b",
            
            # Llama系列
            "llama3.2:1b",
            "llama3.2:3b",
            "llama3.1:8b",
            "llama3.1:70b",
            
            # GPT-OSS
            "gpt-oss:latest",
            "gpt-oss:small",
            "gpt-oss:medium",
            "gpt-oss:large",
            
            # DeepSeek系列
            "deepseek-r1:7b",
            "deepseek-r1:14b",
            "deepseek-r1:32b",
            "deepseek-r1:70b",
            
            # その他
            "mixtral:8x7b",
            "phi3:mini",
            "phi3:medium",
            "solar:latest",
            "yi:34b",
            "command-r:latest",
            "codellama:13b",
            "mistral:7b",
            "neural-chat:7b",
            "starling-lm:7b",
            "vicuna:13b"
        ]
    
    def get_sorted_models(self, available_models: List[str]) -> List[str]:
        """
        モデルを推奨順にソート
        
        Args:
            available_models: 利用可能なモデルリスト
            
        Returns:
            ソート済みモデルリスト
        """
        # 推奨モデルの優先順位
        priority_order = [
            # 日本語最適化
            "qwen2.5:7b",
            "qwen2.5:14b",
            "qwen2.5:32b",
            
            # 一般用途
            "gemma2:9b",
            "gemma3:12b",
            "gemma2:27b",
            
            # 軽量
            "llama3.2:3b",
            "llama3.1:8b",
            
            # 特殊用途
            "gpt-oss:latest",
            "deepseek-r1:7b",
            "deepseek-r1:14b",
            
            # その他
            "mixtral:8x7b",
            "phi3:medium",
            "command-r:latest"
        ]
        
        # 優先モデルと利用可能モデルの交差
        priority_available = []
        for model in priority_order:
            if model in available_models:
                priority_available.append(model)
        
        # その他のモデル
        other_models = sorted([
            m for m in available_models 
            if m not in priority_available
        ])
        
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
        モデルと用途に応じた推奨温度を取得
        
        Args:
            model_name: モデル名
            use_case: 用途（dialogue, director等）
            
        Returns:
            推奨温度
        """
        temp_config = self.model_config.get("temperature_recommendations", {})
        
        if use_case == "director":
            return temp_config.get("director_judgment", {}).get("default", 0.3)
        elif use_case == "creative":
            return temp_config.get("creative_dialogue", {}).get("default", 0.7)
        else:
            return temp_config.get("consistent_character", {}).get("default", 0.6)
    
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
            self.client.pull(model_name)
            return True
            
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