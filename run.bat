@echo off
echo ==============================================
echo  Starting Smart Placement Intelligence System
echo ==============================================

:: Check if virtual environment exists, create if missing
if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Install required packages
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt

:: Open browser after a short delay
echo.
echo ==============================================
echo  App running at: http://127.0.0.1:5000
echo  Press Ctrl+C to stop the server
echo ==============================================
start http://127.0.0.1:5000

:: Run the Flask server
python app.py
pause
