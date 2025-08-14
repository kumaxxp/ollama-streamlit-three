#!/bin/bash
# Advanced Dialogue System 起動スクリプト

echo "🎭 Advanced Dialogue System 起動中..."

# Python警告を制御
export PYTHONWARNINGS="ignore::RuntimeWarning"

# asyncio デバッグモードを無効化
export PYTHONASYNCIODEBUG=0

# Streamlit設定
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# ディレクトリ作成（必要な場合）
mkdir -p .streamlit
mkdir -p data/dialogues
mkdir -p logs

# Ollamaが起動しているか確認
if ! ollama list > /dev/null 2>&1; then
    echo "⚠️  Ollamaが起動していません。起動してください："
    echo "   ollama serve"
    exit 1
fi

echo "✅ Ollama接続確認済み"

# Streamlitアプリを起動
echo "🚀 アプリケーションを起動します..."
echo "   URL: http://localhost:8501"
echo ""

# ログを抑制しながら起動
streamlit run app/pages/03_Advanced_Dialogue.py \
    --server.port 8501 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --logger.level warning \
    2>&1 | grep -v "RuntimeWarning" | grep -v "tracemalloc"