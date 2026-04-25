@echo off
setlocal

if not exist venv (
  python -m venv venv
)

call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Base environment setup complete.
echo Optional deep-learning packages:
echo   pip install -r requirements_dl.txt

echo.
echo Run full pipeline:
echo   python main.py --quick

echo Run interactive predictor:
echo   python predict.py

endlocal
