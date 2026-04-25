# SMS Spam Project

A modular SMS spam detection project comparing five approaches:

1. Naive Bayes
2. SVM
3. CNN
4. LSTM
5. Zero-shot LLM (Ollama by default)

## Project Structure

```text
sms_spam_project/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocess.py
│   ├── traditional_models.py
│   ├── deep_models.py
│   ├── llm_classifier.py
│   ├── evaluate.py
│   └── visualise.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_traditional.ipynb
│   ├── 03_deep_learning.ipynb
│   ├── 04_llm.ipynb
│   └── 05_comparison.ipynb
├── data/
│   └── README.md
├── models/
│   └── README.md
├── results/
│   └── README.md
├── tests/
│   ├── test_preprocess.py
│   └── test_model_loading.py
├── main.py
├── predict.py
├── requirements.txt
├── requirements_dl.txt
├── setup.bat
├── setup.sh
└── .gitignore
```

## Quick Start

### Windows

```powershell
setup.bat
```

### macOS/Linux

```bash
chmod +x setup.sh
./setup.sh
```

## Install Dependencies Manually

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

Optional deep-learning dependencies:

```bash
pip install -r requirements_dl.txt
```

## Dataset

1. Download from UCI: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
2. Place file at `data/SMSSpamCollection`

## Run Full Pipeline

```bash
python main.py --quick
```

Artifacts are generated under `results/` and model files under `models/`.

## Predict New Message

Interactive mode:

```bash
python predict.py
```

One-shot mode:

```bash
python predict.py --message "Win a free ticket now"
```

## LLM Backend

`main.py` uses Ollama by default via `src/llm_classifier.py`.

To use OpenAI in custom code, set:

```bash
OPENAI_API_KEY=your_key_here
```

## Tests

```bash
pytest -q
```

## Notes

- `data/` raw dataset and `models/` artifacts are ignored by git.
- Legacy scripts (`compare_models.py`, `predict_new.py`, `generate_visuals.py`, `spam_detector_complete.py`) are thin wrappers to the canonical `main.py` and `predict.py` entry points.
