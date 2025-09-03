#!/bin/bash
# Advanced Dialogue System 起動スクリプト（conda対応・安定化）

echo "🎭 Advanced Dialogue System 起動中..."

# conda 環境があれば自動有効化（ollama-chat 優先）
if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1090
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    if conda env list | grep -q "^ollama-chat\s"; then
        conda activate ollama-chat >/dev/null 2>&1 || true
        echo "🧪 Using conda env: $(python -V 2>/dev/null || echo unknown)"
    fi
fi

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

# 直接起動（フィルタで重要ログを隠さない）
python -m streamlit run app/pages/03_Advanced_Dialogue_Refactored.py \
    --server.port 8501 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --logger.level info