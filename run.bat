@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found - creating one and installing dependencies...
    python -m venv .venv
    ".venv\Scripts\python.exe" -m pip install -q -r requirements.txt -r requirements-app.txt
)

echo Starting HS Code Classifier...
echo Your browser should open automatically in a few seconds.
echo Leave THIS WINDOW OPEN while using the app - closing it stops the app.
echo.

".venv\Scripts\python.exe" -m streamlit run app.py

echo.
echo HS Code Classifier has stopped.
pause
