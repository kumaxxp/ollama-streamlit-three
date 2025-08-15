#!/usr/bin/env python3
"""
Ollamaモデル確認・ダウンロードスクリプト
Advanced Dialogue System用
"""

import sys
import ollama
from typing import List, Set
from app.core.model_utils import ModelManager

def check_ollama_connection() -> bool:
    """Ollama接続確認"""
    try:
        client = ollama.Client()
        client.list()
        print("✅ Ollamaに接続できました")
        return True
    except Exception as e:
        print(f"❌ Ollamaに接続できません: {e}")
        print("\n以下を確認してください:")
        print("1. Ollamaがインストールされているか: https://ollama.ai")
        print("2. Ollamaが起動しているか: ollama serve")
        return False

def list_available_models() -> List[str]:
    """利用可能なモデル一覧を表示"""
    manager = ModelManager()
    models = manager.get_available_models()
    
    if not models:
        print("⚠️ インストール済みモデルがありません")
        return []
    
    print(f"\n📦 インストール済みモデル ({len(models)}個):")
    print("-" * 50)
    
    # カテゴリ別に分類
    categories = {
        "Qwen系": [],
        "Gemma系": [],
        "Llama系": [],
        "GPT-OSS": [],
        "DeepSeek系": [],
        "その他": []
    }
    
    for model in models:
        if "qwen" in model.lower():
            categories["Qwen系"].append(model)
        elif "gemma" in model.lower():
            categories["Gemma系"].append(model)
        elif "llama" in model.lower():
            categories["Llama系"].append(model)
        elif "gpt" in model.lower():
            categories["GPT-OSS"].append(model)
        elif "deepseek" in model.lower():
            categories["DeepSeek系"].append(model)
        else:
            categories["その他"].append(model)
    
    for category, model_list in categories.items():
        if model_list:
            print(f"\n{category}:")
            for model in model_list:
                print(f"  • {model}")
    
    return models

def check_recommended_models(available_models: Set[str]) -> None:
    """推奨モデルの確認"""
    print("\n🎯 本番環境推奨モデルの状態:")
    print("-" * 50)
    
    # 本番環境で採用するモデル
    production_models = [
        ("qwen2.5:7b-instruct-q4_K_M", "日本語対話エージェント推奨", "4.7GB", "★★★★★"),
        ("gemma3:12b", "高品質エージェント", "8.1GB", "★★★★☆"),
        ("gpt-oss:20b", "創造的対話エージェント", "13GB", "★★★☆☆"),
        ("gemma3:4b", "Director推奨（高速判断）", "3.3GB", "★★★★☆"),
        ("qwen:7b", "フォールバック用", "4.5GB", "★★★☆☆")
    ]
    
    print(f"{'モデル名':<30} {'用途':<25} {'サイズ':<8} {'推奨度'}")
    print("-" * 80)
    
    missing = []
    for model_name, description, size, rating in production_models:
        if model_name in available_models:
            print(f"✅ {model_name:<30} {description:<25} {size:<8} {rating}")
        else:
            print(f"❌ {model_name:<30} {description:<25} {size:<8} {rating}")
            missing.append(model_name)
    
    if missing:
        print("\n💡 推奨モデルのインストールコマンド:")
        print("-" * 50)
        for model in missing:
            print(f"   ollama pull {model}")
        
        print("\n📌 特に重要:")
        print("   ollama pull qwen2.5:7b-instruct-q4_K_M  # エージェント用")
        print("   ollama pull gemma3:4b                   # Director用")

def suggest_models_by_vram() -> None:
    """VRAM容量別の推奨"""
    print("\n💻 VRAM容量別の推奨モデル:")
    print("-" * 50)
    
    suggestions = {
        "3-4GB": ["llama3.2:3b", "phi3:mini"],
        "6-8GB": ["qwen2.5:7b", "gemma2:9b", "llama3.1:8b"],
        "12-16GB": ["qwen2.5:14b", "gemma3:12b", "deepseek-r1:14b"],
        "24GB以上": ["qwen2.5:32b", "gemma2:27b", "mixtral:8x7b"]
    }
    
    for vram, models in suggestions.items():
        print(f"\nVRAM {vram}:")
        for model in models:
            print(f"  • {model}")

def download_model(model_name: str) -> bool:
    """モデルをダウンロード"""
    print(f"\n📥 {model_name} をダウンロード中...")
    try:
        client = ollama.Client()
        
        # プログレス表示のためのコールバック
        for progress in client.pull(model_name, stream=True):
            status = progress.get('status', '')
            if 'total' in progress and 'completed' in progress:
                total = progress['total']
                completed = progress['completed']
                percent = (completed / total) * 100 if total > 0 else 0
                print(f"\r{status}: {percent:.1f}%", end='')
            else:
                print(f"\r{status}", end='')
        
        print(f"\n✅ {model_name} のダウンロードが完了しました")
        return True
        
    except Exception as e:
        print(f"\n❌ ダウンロードエラー: {e}")
        return False

def interactive_download() -> None:
    """対話的なモデルダウンロード"""
    print("\n📦 本番環境推奨モデルをダウンロードしますか？")
    print("1. qwen2.5:7b-instruct-q4_K_M (エージェント推奨、4.7GB)")
    print("2. gemma3:4b (Director推奨、3.3GB)")
    print("3. gemma3:12b (高品質エージェント、8.1GB)")
    print("4. gpt-oss:20b (創造的対話、13GB)")
    print("5. 全推奨モデルをインストール")
    print("6. カスタム入力")
    print("0. スキップ")
    
    choice = input("\n選択 (0-6): ").strip()
    
    model_map = {
        "1": "qwen2.5:7b-instruct-q4_K_M",
        "2": "gemma3:4b",
        "3": "gemma3:12b",
        "4": "gpt-oss:20b"
    }
    
    if choice in model_map:
        download_model(model_map[choice])
    elif choice == "5":
        # 全推奨モデルをインストール
        models_to_install = [
            "qwen2.5:7b-instruct-q4_K_M",
            "gemma3:4b",
            "gemma3:12b",
            "gpt-oss:20b",
            "qwen:7b"
        ]
        for model in models_to_install:
            download_model(model)
    elif choice == "6":
        custom_model = input("モデル名を入力: ").strip()
        if custom_model:
            download_model(custom_model)
    elif choice == "0":
        print("スキップしました")

def main():
    """メイン処理"""
    print("=" * 60)
    print("🎭 Advanced Dialogue System - モデル確認ツール")
    print("=" * 60)
    
    # Ollama接続確認
    if not check_ollama_connection():
        sys.exit(1)
    
    # モデル一覧表示
    available = list_available_models()
    available_set = set(available)
    
    # 推奨モデル確認
    check_recommended_models(available_set)
    
    # VRAM別推奨
    suggest_models_by_vram()
    
    # 対話的ダウンロード
    if len(available) == 0:
        print("\n⚠️ モデルがインストールされていません")
        interactive_download()
    else:
        choice = input("\n追加でモデルをダウンロードしますか？ (y/n): ").strip().lower()
        if choice == 'y':
            interactive_download()
    
    print("\n✨ 確認完了！")
    print("Streamlitアプリを起動: streamlit run app/pages/03_Advanced_Dialogue_Refactored.py")

if __name__ == "__main__":
    main()