@echo off
title FedShield — Starting...
color 0A

echo.
echo  ============================================
echo   FedShield — Launching All Services
echo  ============================================
echo.

cd /d C:\Users\megha\OneDrive\Desktop\fedshield

echo  [1/3] Starting FastAPI backend...
start "FedShield API" cmd /k "venv\Scripts\activate && uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload"

timeout /t 2 /nobreak > nul

echo  [2/3] Starting React frontend...
start "FedShield Web" cmd /k "cd web && npm run dev"

timeout /t 3 /nobreak > nul

echo  [3/3] Opening browser...
start http://localhost:3000

echo.
echo  ============================================
echo   FedShield is running!
echo.
echo   Web Dashboard : http://localhost:3000
echo   API           : http://localhost:8000
echo   Streamlit     : http://localhost:8501
echo  ============================================
echo.
echo  To start live capture (Admin terminal):
echo    python live_capture.py
echo.
pause