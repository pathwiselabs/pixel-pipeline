@echo off
setlocal EnableDelayedExpansion

:: ────────────────────────────────────────────────────────────────────────────
::  Pixel Pipeline - Detailed Startup Script
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

:: Check Python version
echo [%time%] Checking Python version...
python --version > temp_ver.txt
set /p PYTHON_VERSION=<temp_ver.txt
del temp_ver.txt
echo [%time%] [INFO] %PYTHON_VERSION%

:: Check PyTorch version
echo [%time%] Checking PyTorch version...
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" > torch_ver.txt 2>nul
if errorlevel 1 (
    echo [%time%] [WARNING] PyTorch not found in environment. Some features may not work properly.
) else (
    set /p TORCH_VERSION=<torch_ver.txt
    echo [%time%] [INFO] %TORCH_VERSION%
)
del torch_ver.txt 2>nul

:: Check CUDA availability
echo [%time%] Checking CUDA availability...
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"Not available\"}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU found\"}')" > cuda_info.txt 2>nul
if errorlevel 1 (
    echo [%time%] [WARNING] Could not determine CUDA availability.
) else (
    type cuda_info.txt | findstr /R "." > nul
    if errorlevel 1 (
        echo [%time%] [WARNING] CUDA information unavailable.
    ) else (
        for /f "tokens=*" %%a in (cuda_info.txt) do (
            echo [%time%] [INFO] %%a
        )
    )
)
del cuda_info.txt 2>nul

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