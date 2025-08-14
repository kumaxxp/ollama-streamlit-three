#!/bin/bash

# Advanced Dialogue System - Setup and Run Script

echo "🚀 Advanced Dialogue System Launcher"
echo "===================================="

# カラー設定
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Conda環境のチェック
check_conda_env() {
    if conda info --envs | grep -q "ollama-chat"; then
        echo -e "${GREEN}✓ Conda環境 'ollama-chat' が見つかりました${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠ Conda環境 'ollama-chat' が見つかりません${NC}"
        return 1
    fi
}

# Ollamaサービスのチェック
check_ollama() {
    if systemctl is-active --quiet ollama; then
        echo -e "${GREEN}✓ Ollamaサービスが起動しています${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠ Ollamaサービスが起動していません${NC}"
        echo "起動しますか？ (y/n)"
        read -r response
        if [[ "$response" == "y" ]]; then
            sudo systemctl start ollama
            sleep 2
            if systemctl is-active --quiet ollama; then
                echo -e "${GREEN}✓ Ollamaサービスを起動しました${NC}"
                return 0
            else
                echo -e "${RED}✗ Ollamaサービスの起動に失敗しました${NC}"
                return 1
            fi
        fi
        return 1
    fi
}

# モデルのチェック
check_models() {
    echo "利用可能なモデルをチェック中..."
    
    # 推奨モデルリスト
    recommended_models=("qwen2.5:7b" "gemma3:4b")
    missing_models=()
    
    for model in "${recommended_models[@]}"; do
        if ollama list | grep -q "$model"; then
            echo -e "${GREEN}✓ $model${NC}"
        else
            echo -e "${YELLOW}⚠ $model が見つかりません${NC}"
            missing_models+=("$model")
        fi
    done
    
    if [ ${#missing_models[@]} -gt 0 ]; then
        echo "不足しているモデルをダウンロードしますか？ (y/n)"
        read -r response
        if [[ "$response" == "y" ]]; then
            for model in "${missing_models[@]}"; do
                echo "Downloading $model..."
                ollama pull "$model"
            done
        fi
    fi
}

# ディレクトリ構造の確認と作成
setup_directories() {
    echo "ディレクトリ構造を確認中..."
    
    directories=("config" "app/core" "app/pages" "data/dialogues")
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            echo -e "${GREEN}✓ $dir を作成しました${NC}"
        else
            echo -e "${GREEN}✓ $dir${NC}"
        fi
    done
}

# 設定ファイルの確認
check_config_files() {
    echo "設定ファイルを確認中..."
    
    config_files=("config/characters.json" "config/strategies.json" "config/prompt_templates.json")
    missing_configs=()
    
    for file in "${config_files[@]}"; do
        if [ -f "$file" ]; then
            echo -e "${GREEN}✓ $file${NC}"
        else
            echo -e "${RED}✗ $file が見つかりません${NC}"
            missing_configs+=("$file")
        fi
    done
    
    if [ ${#missing_configs[@]} -gt 0 ]; then
        echo -e "${RED}設定ファイルが不足しています。GitHubから最新版を取得してください。${NC}"
        return 1
    fi
    return 0
}

# メインメニュー
show_menu() {
    echo ""
    echo "起動するページを選択してください:"
    echo "1) 01_Simple_Chat - シンプルチャット"
    echo "2) 03_Advanced_Dialogue - Advanced対話システム（推奨）"
    echo "3) すべてのページ（サイドバーで切り替え可能）"
    echo "q) 終了"
    echo ""
    echo -n "選択 [1-3, q]: "
    read -r choice
    
    case $choice in
        1)
            if [ -f "app/pages/01_Simple_Chat.py" ]; then
                streamlit run app/pages/01_Simple_Chat.py
            elif [ -f "app/simple_chat.py" ]; then
                streamlit run app/simple_chat.py
            else
                echo -e "${RED}Simple Chatが見つかりません${NC}"
            fi
            ;;
        2)
            if [ -f "app/pages/03_Advanced_Dialogue.py" ]; then
                streamlit run app/pages/03_Advanced_Dialogue.py
            else
                echo -e "${RED}Advanced Dialogueが見つかりません${NC}"
            fi
            ;;
        3)
            # メインファイルを探す
            if [ -f "app/pages/01_Simple_Chat.py" ]; then
                streamlit run app/pages/01_Simple_Chat.py
            elif [ -f "app/pages/03_Advanced_Dialogue.py" ]; then
                streamlit run app/pages/03_Advanced_Dialogue.py
            else
                echo -e "${RED}起動可能なページが見つかりません${NC}"
            fi
            ;;
        q)
            echo "終了します"
            exit 0
            ;;
        *)
            echo -e "${RED}無効な選択です${NC}"
            show_menu
            ;;
    esac
}

# メイン処理
main() {
    echo ""
    
    # 環境チェック
    if ! check_conda_env; then
        echo -e "${RED}Conda環境をセットアップしてください:${NC}"
        echo "conda create -n ollama-chat python=3.11"
        echo "conda activate ollama-chat"
        echo "pip install -r requirements.txt"
        exit 1
    fi
    
    # Conda環境をアクティベート
    echo "Conda環境をアクティベート中..."
    eval "$(conda shell.bash hook)"
    conda activate ollama-chat
    
    # Ollamaチェック
    if ! check_ollama; then
        echo -e "${YELLOW}Ollamaなしで続行します（エラーが発生する可能性があります）${NC}"
    fi
    
    # モデルチェック
    check_models
    
    # ディレクトリセットアップ
    setup_directories
    
    # 設定ファイルチェック
    if ! check_config_files; then
        echo -e "${RED}設定ファイルエラーにより終了します${NC}"
        exit 1
    fi
    
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}セットアップ完了！${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # メニュー表示
    show_menu
}

# スクリプト実行
main