@echo off
echo Starting GPIO Extractor...
echo.
echo ============================================================
echo  GPIO Extractor - Board Schematic Analyzer
echo  Open browser at: http://127.0.0.1:5000
echo ============================================================
echo.
cd /d "%~dp0"
.venv\Scripts\python.exe app.py
pause
