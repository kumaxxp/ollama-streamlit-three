#!/bin/bash
# Advanced Dialogue System - 推奨モデルインストールスクリプト

echo "=========================================="
echo "🎭 Advanced Dialogue System"
echo "   推奨モデルインストーラー"
echo "=========================================="
echo ""

# Ollamaの起動確認
check_ollama() {
    if ! ollama list > /dev/null 2>&1; then
        echo "❌ Ollamaが起動していません"
        echo "以下のコマンドで起動してください:"
        echo "  ollama serve"
        exit 1
    fi
    echo "✅ Ollamaが起動しています"
}

# モデルのインストール関数
install_model() {
    local model=$1
    local description=$2
    
    echo ""
    echo "📦 インストール中: $model"
    echo "   説明: $description"
    
    if ollama list | grep -q "$model"; then
        echo "   ✅ 既にインストール済み"
    else
        echo "   ⏳ ダウンロード開始..."
        if ollama pull "$model"; then
            echo "   ✅ インストール完了"
        else
            echo "   ❌ インストール失敗"
            return 1
        fi
    fi
}

# メイン処理
main() {
    check_ollama
    
    echo ""
    echo "インストールレベルを選択してください:"
    echo "1) 最小構成 (VRAM 6GB) - 必須モデルのみ"
    echo "2) 標準構成 (VRAM 10GB) - 推奨モデル"
    echo "3) フル構成 (VRAM 16GB+) - 全モデル"
    echo "4) カスタム選択"
    echo "0) 終了"
    echo ""
    read -p "選択 (0-4): " choice
    
    case $choice in
        1)
            echo ""
            echo "📋 最小構成をインストールします"
            install_model "qwen2.5:7b-instruct-q4_K_M" "日本語対話エージェント（必須）"
            install_model "gemma3:4b" "Director AI（必須）"
            ;;
        2)
            echo ""
            echo "📋 標準構成をインストールします"
            install_model "qwen2.5:7b-instruct-q4_K_M" "日本語対話エージェント（必須）"
            install_model "gemma3:4b" "Director AI（必須）"
            install_model "gemma3:12b" "高品質エージェント"
            ;;
        3)
            echo ""
            echo "📋 フル構成をインストールします"
            install_model "qwen2.5:7b-instruct-q4_K_M" "日本語対話エージェント（必須）"
            install_model "gemma3:4b" "Director AI（必須）"
            install_model "gemma3:12b" "高品質エージェント"
            install_model "gpt-oss:20b" "創造的対話エージェント"
            install_model "qwen:7b" "フォールバック用"
            ;;
        4)
            echo ""
            echo "📋 カスタム選択"
            echo ""
            echo "インストールするモデルを選択してください（複数選択可）:"
            echo "a) qwen2.5:7b-instruct-q4_K_M (4.7GB)"
            echo "b) gemma3:4b (3.3GB)"
            echo "c) gemma3:12b (8.1GB)"
            echo "d) gpt-oss:20b (13GB)"
            echo "e) qwen:7b (4.5GB)"
            echo ""
            read -p "選択（例: abc）: " models
            
            [[ $models == *a* ]] && install_model "qwen2.5:7b-instruct-q4_K_M" "日本語対話エージェント"
            [[ $models == *b* ]] && install_model "gemma3:4b" "Director AI"
            [[ $models == *c* ]] && install_model "gemma3:12b" "高品質エージェント"
            [[ $models == *d* ]] && install_model "gpt-oss:20b" "創造的対話"
            [[ $models == *e* ]] && install_model "qwen:7b" "フォールバック"
            ;;
        0)
            echo "終了します"
            exit 0
            ;;
        *)
            echo "無効な選択です"
            exit 1
            ;;
    esac
    
    echo ""
    echo "=========================================="
    echo "✨ インストール処理が完了しました"
    echo ""
    echo "インストール済みモデル:"
    ollama list | grep -E "qwen2.5:7b-instruct-q4_K_M|gemma3:4b|gemma3:12b|gpt-oss:20b|qwen:7b"
    echo ""
    echo "次のステップ:"
    echo "  python check_models.py  # モデル確認"
    echo "  streamlit run app/pages/03_Advanced_Dialogue.py  # アプリ起動"
    echo "=========================================="
}

# スクリプト実行
main