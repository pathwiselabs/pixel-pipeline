@echo off
setlocal EnableDelayedExpansion

:: ────────────────────────────────────────────────────────────────────────────
::  Pixel Pipeline - Informative Startup Script
::  This script helps launch the Gradio-based Pixel Pipeline application.
:: ────────────────────────────────────────────────────────────────────────────

echo.
echo ===============================================
echo Pixel Pipeline Launcher
echo Started at %date% %time%
echo ===============================================
echo.

:: Go to the batch file's directory
echo [%time%] Heading to Pixel Pipeline directory...
cd /d "%~dp0" || (
    echo [%time%] [ERROR] Failed: cannot access directory %~dp0
    pause
    exit /b 1
)
echo [%time%] [OK] Working directory set.

:: Activate Python virtual environment
echo [%time%] Activating Python virtual environment...
call "%~dp0venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [%time%] [ERROR] Virtual environment activation failed. Make sure 'venv' exists.
    pause
    exit /b 1
)
echo [%time%] [OK] Python virtual environment activated.

:: Check dependencies informatively (Optional nice touch)
echo [%time%] Checking main dependencies...
python -c "import gradio" >nul 2>&1
if errorlevel 1 (
    echo [%time%] [WARNING] Gradio not found in environment. Consider running: pip install -r requirements.txt
    pause
    exit /b 1
) else (
    echo [%time%] [OK] Gradio dependency found.
)

:: Launch Gradio app clearly within the same command window
echo [%time%] Launching Pixel Pipeline Application...
echo [%time%] Starting app.py... Please be patient; it may take several moments to start.
echo.

python "%~dp0app.py"

:: After the app terminates (e.g., CTRL+C or app exits normally)
if errorlevel 1 (
    echo.
    echo [%time%] [ERROR] Pixel Pipeline exited with errors. Review log output above.
) else (
    echo.
    echo [%time%] Pixel Pipeline closed without errors.
)

echo ===============================================
echo [%time%] Session completed. You may close this window.
echo ===============================================
pause

endlocal