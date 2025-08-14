"""
Advanced Dialogue System - Application Package
警告制御とログ設定
"""

import warnings
import logging
import os

# RuntimeWarningを抑制
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*coroutine.*was never awaited")
warnings.filterwarnings("ignore", message=".*tracemalloc.*")

# asyncio関連の警告を抑制
import asyncio
if hasattr(asyncio, 'set_event_loop_policy'):
    if os.name == 'nt':  # Windows
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ログレベル設定
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Streamlit特有の警告を抑制
logging.getLogger('streamlit.runtime.caching').setLevel(logging.ERROR)
logging.getLogger('streamlit.runtime.legacy_caching').setLevel(logging.ERROR)

# バージョン情報
__version__ = "2.0.0"
__author__ = "Advanced Dialogue System Team"