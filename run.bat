@echo off
cd /d "%~dp0"

REM Backend (FastAPI): use venv-gpu if present, else venv
set "VENV=venv-gpu"
if not exist "backend\venv-gpu\Scripts\activate.bat" set "VENV=venv"

start "Backend" cmd /k "cd /d ""%~dp0backend"" && call .\%VENV%\Scripts\activate && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"

REM Wait for backend to bind port 8000 before frontend hits /api
timeout /t 4 /nobreak >nul

REM Frontend (Vite/React)
start "Frontend" cmd /k "cd /d ""%~dp0frontend"" && npm run dev -- --host 0.0.0.0 --port 5173"

pause
