@echo off
echo Starting Image Tagging Application...
cd /d G:\AI\batch-labels
call venv\Scripts\activate
python app.py
pause