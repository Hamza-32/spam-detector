#!/bin/bash
# Run this ONCE to set up your environment
echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating..."
source venv/bin/activate

echo "Installing packages..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo "To run the project:"
echo "  source venv/bin/activate"
echo "  python spam_detector_complete.py"
