@echo off
echo ============================================
echo Starting RAG Server
echo ============================================
echo.

REM Activate Python environment if needed
REM call venv\Scripts\activate.bat

REM Run server (unbuffered stdout for immediate logs)
python -u serve_rag.py

pause
