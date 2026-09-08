@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo   CUSTOMER CHURN PREDICTION & RETENTION SYSTEM
echo ===================================================
echo.

:: 1. Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in PATH.
    echo Please install Python 3.9+ from https://www.python.org/
    pause
    exit /b 1
)

echo [OK] Python detected.

:: 2. Check or create virtual environment
if not exist "venv\Scripts\activate.bat" (
    echo [INFO] Creating virtual environment (venv)...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
)

:: 3. Activate virtual environment
call venv\Scripts\activate.bat

:: 4. Install dependencies if needed
echo [INFO] Checking dependencies...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [WARNING] Some dependencies had warnings during install. Proceeding...
)

:: 5. Start main application
echo.
echo [INFO] Starting Flask Application...
python app.py

pause
