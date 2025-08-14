@echo off
chcp 65001 > nul
title Advanced Dialogue System - モデルインストーラー

echo ==========================================
echo 🎭 Advanced Dialogue System
echo    推奨モデルインストーラー
echo ==========================================
echo.

:: Ollamaの起動確認
ollama list > nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Ollamaが起動していません
    echo 以下のコマンドで起動してください:
    echo   ollama serve
    pause
    exit /b 1
)
echo ✅ Ollamaが起動しています
echo.

echo インストールレベルを選択してください:
echo 1) 最小構成 (VRAM 6GB) - 必須モデルのみ
echo 2) 標準構成 (VRAM 10GB) - 推奨モデル
echo 3) フル構成 (VRAM 16GB+) - 全モデル
echo 4) 個別インストール
echo 0) 終了
echo.
set /p choice="選択 (0-4): "

if "%choice%"=="1" goto minimal
if "%choice%"=="2" goto standard
if "%choice%"=="3" goto full
if "%choice%"=="4" goto custom
if "%choice%"=="0" goto end
goto invalid

:minimal
echo.
echo 📋 最小構成をインストールします
echo.
echo [1/2] qwen2.5:7b-instruct-q4_K_M (日本語対話エージェント)
ollama pull qwen2.5:7b-instruct-q4_K_M
echo.
echo [2/2] gemma3:4b (Director AI)
ollama pull gemma3:4b
goto complete

:standard
echo.
echo 📋 標準構成をインストールします
echo.
echo [1/3] qwen2.5:7b-instruct-q4_K_M (日本語対話エージェント)
ollama pull qwen2.5:7b-instruct-q4_K_M
echo.
echo [2/3] gemma3:4b (Director AI)
ollama pull gemma3:4b
echo.
echo [3/3] gemma3:12b (高品質エージェント)
ollama pull gemma3:12b
goto complete

:full
echo.
echo 📋 フル構成をインストールします
echo.
echo [1/5] qwen2.5:7b-instruct-q4_K_M (日本語対話エージェント)
ollama pull qwen2.5:7b-instruct-q4_K_M
echo.
echo [2/5] gemma3:4b (Director AI)
ollama pull gemma3:4b
echo.
echo [3/5] gemma3:12b (高品質エージェント)
ollama pull gemma3:12b
echo.
echo [4/5] gpt-oss:20b (創造的対話)
ollama pull gpt-oss:20b
echo.
echo [5/5] qwen:7b (フォールバック)
ollama pull qwen:7b
goto complete

:custom
echo.
echo 📋 個別インストール
echo.
echo インストールするモデルを選択してください:
echo 1) qwen2.5:7b-instruct-q4_K_M (4.7GB) - エージェント推奨
echo 2) gemma3:4b (3.3GB) - Director推奨
echo 3) gemma3:12b (8.1GB) - 高品質エージェント
echo 4) gpt-oss:20b (13GB) - 創造的対話
echo 5) qwen:7b (4.5GB) - フォールバック
echo 0) 戻る
echo.

:custom_loop
set /p model_choice="モデル番号を入力 (0で終了): "

if "%model_choice%"=="0" goto complete
if "%model_choice%"=="1" (
    echo インストール中: qwen2.5:7b-instruct-q4_K_M
    ollama pull qwen2.5:7b-instruct-q4_K_M
    goto custom_loop
)
if "%model_choice%"=="2" (
    echo インストール中: gemma3:4b
    ollama pull gemma3:4b
    goto custom_loop
)
if "%model_choice%"=="3" (
    echo インストール中: gemma3:12b
    ollama pull gemma3:12b
    goto custom_loop
)
if "%model_choice%"=="4" (
    echo インストール中: gpt-oss:20b
    ollama pull gpt-oss:20b
    goto custom_loop
)
if "%model_choice%"=="5" (
    echo インストール中: qwen:7b
    ollama pull qwen:7b
    goto custom_loop
)
echo 無効な選択です
goto custom_loop

:invalid
echo 無効な選択です
pause
exit /b 1

:complete
echo.
echo ==========================================
echo ✨ インストール処理が完了しました
echo.
echo インストール済みモデル:
ollama list
echo.
echo 次のステップ:
echo   python check_models.py  (モデル確認)
echo   streamlit run app/pages/03_Advanced_Dialogue.py  (アプリ起動)
echo ==========================================
pause
goto end

:end
exit /b 0