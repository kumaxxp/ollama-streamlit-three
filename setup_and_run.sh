#!/bin/bash

# Ollama + Streamlit チャットシステム セットアップスクリプト

echo "🚀 Ollama Chat システムセットアップ開始"

# 1. Ollamaの状態確認
echo "📡 Ollamaサーバーの状態を確認中..."
if systemctl is-active --quiet ollama; then
    echo "✅ Ollamaは起動しています"
else
    echo "⚠️  Ollamaが起動していません。起動を試みます..."
    sudo systemctl start ollama
    sleep 3
    if systemctl is-active --quiet ollama; then
        echo "✅ Ollamaを起動しました"
    else
        echo "❌ Ollama起動失敗。手動で起動してください："
        echo "   sudo systemctl start ollama"
        exit 1
    fi
fi

# 2. Qwen2.5-7bモデルの確認とダウンロード
echo "🤖 Qwen2.5-7bモデルを確認中..."
if ollama list | grep -q "qwen2.5:7b"; then
    echo "✅ Qwen2.5-7bは既にインストール済み"
else
    echo "📥 Qwen2.5-7bをダウンロード中..."
    echo "   (約4.5GB、時間がかかります)"
    ollama pull qwen2.5:7b
    if [ $? -eq 0 ]; then
        echo "✅ Qwen2.5-7bのダウンロード完了"
    else
        echo "⚠️  Qwen2.5-7bのダウンロードに失敗しました"
        echo "   他のモデルで続行します"
    fi
fi

# 3. Conda環境のセットアップ
echo "🐍 Conda環境をセットアップ中..."
if conda env list | grep -q "ollama-chat"; then
    echo "✅ ollama-chat環境は既に存在します"
    conda activate ollama-chat 2>/dev/null || source activate ollama-chat
else
    echo "📦 新しいConda環境を作成中..."
    conda env create -f environment.yml
    if [ $? -eq 0 ]; then
        echo "✅ Conda環境作成完了"
        conda activate ollama-chat 2>/dev/null || source activate ollama-chat
    else
        echo "❌ Conda環境作成失敗"
        echo "   手動でセットアップしてください："
        echo "   conda create -n ollama-chat python=3.11"
        echo "   conda activate ollama-chat"
        echo "   pip install streamlit ollama"
        exit 1
    fi
fi

# 4. 利用可能なモデル一覧表示
echo ""
echo "📋 利用可能なモデル:"
ollama list

# 5. Streamlitアプリケーション起動
echo ""
echo "🌐 Streamlitアプリケーションを起動中..."
echo "   URL: http://localhost:8501"
echo ""
echo "終了するには Ctrl+C を押してください"
echo "=" * 50

# Streamlit起動
streamlit run app/simple_chat.py \
    --server.port 8501 \
    --server.address localhost \
    --browser.gatherUsageStats false