@echo off
echo Creating virtual environment...
python -m venv venv

echo Activating...
call venv\Scripts\activate.bat

echo Installing packages...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Setup complete!
echo To run the project:
echo   venv\Scripts\activate
echo   python spam_detector_complete.py
