@echo off
title Traffic Monitor – FastAPI Backend
echo ═══════════════════════════════════════════
echo   Traffic Monitor FastAPI Backend v2
echo   API docs: http://localhost:8000/docs
echo ═══════════════════════════════════════════
echo.

if exist "backend\venv\Scripts\activate.bat" (
    echo [INFO] Activating venv...
    call backend\venv\Scripts\activate.bat
)

cd backend
echo [INFO] Starting uvicorn on :8000 ...
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

pause
