@echo off
TITLE CryptoHunter Launcher
color 0A

echo ===================================================
echo     CRYPTOHUNTER AI - AUTOMATIC LAUNCHER
echo ===================================================
echo.

:: 1. Start Backend (API)
echo [1/2] Starting Backend Server (FastAPI)...
start "CryptoHunter API" cmd /k "python main_api.py"

:: Wait a bit for backend to initialize
timeout /t 5 /nobreak >nul

:: 2. Start Frontend (Next.js)
echo [2/2] Starting Web Interface...
cd web_interface
start "CryptoHunter Dashboard" cmd /k "npm run dev"

echo.
echo [SUCCESS] All systems running!
echo Backend:   http://localhost:8000
echo Frontend:  http://localhost:3000
echo.
echo Press any key to exit this launcher (Servers will keep running)...
pause >nul
