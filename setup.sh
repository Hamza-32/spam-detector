#!/bin/bash
set -e

if [ ! -d "venv" ]; then
	python3 -m venv venv
fi

source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Base environment setup complete."
echo "Optional deep-learning packages:"
echo "  pip install -r requirements_dl.txt"
echo ""
echo "Run full pipeline:"
echo "  python main.py --quick"
echo ""
echo "Run interactive predictor:"
echo "  python predict.py"
