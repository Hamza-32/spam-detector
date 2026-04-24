# 📱 SMS Spam Detector — CSE-4218 Project
**Bangladesh University of Professionals**

---

## 👥 Team
| Name | ID |
|------|-----|
| Sadia Iffat | 2252421012 |
| Hamza Bin Arif | 2252421032 |
| Faizah Mehnaz | 2252421080 |
| Raiyan Bin Sarwar | 2252421096 |

---

## 📁 File Structure
```
spam_detector/
├── data/
│   └── SMSSpamCollection       ← put dataset here (download link below)
├── spam_detector.ipynb         ← MAIN FILE: Jupyter Notebook (recommended)
├── spam_detector_complete.py   ← same code as a single Python script
├── predict_new.py              ← interactive classifier for new messages
├── requirements.txt            ← all required libraries
├── setup.sh                    ← Mac/Linux one-click setup
├── setup_windows.bat           ← Windows one-click setup
└── README.md                   ← this file
```

---

## 🚀 Setup Guide (Beginners)

### Step 1 — Install Python
Download Python 3.10+ from https://www.python.org/downloads/
✅ Check "Add Python to PATH" during installation (Windows)

### Step 2 — Download the Dataset
1. Go to: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
2. Click Download
3. Extract the zip file
4. Copy the file named `SMSSpamCollection` into the `data/` folder

### Step 3 — Set Up Environment

**Windows:**
```
Double-click setup_windows.bat
```

**Mac/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

**Manual (any OS):**
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

### Step 4 — Run the Project

**Option A — Jupyter Notebook (recommended for beginners):**
```bash
jupyter notebook
# Opens browser → click spam_detector.ipynb → Run All
```

**Option B — Run as Python script:**
```bash
python spam_detector_complete.py
```

**Option C — Interactive classifier:**
```bash
python predict_new.py
# Type any SMS and get instant prediction!
```

---

## 📊 Expected Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | ~0.975 | ~0.985 | ~0.900 | ~0.940 |
| Logistic Regression | ~0.985 | ~0.990 | ~0.940 | ~0.965 |
| Zero-Shot BART | ~0.900 | ~0.870 | ~0.880 | ~0.875 |

*Results may vary slightly each run.*

---

## 🆓 Free Tools Used
| Tool | Purpose | Cost |
|------|---------|------|
| Python | Programming language | Free |
| scikit-learn | Naive Bayes, Logistic Regression | Free |
| HuggingFace Transformers | Zero-shot LLM (BART) | Free |
| Matplotlib/Seaborn | Charts | Free |
| Jupyter Notebook | Interactive coding | Free |
| UCI SMS Dataset | Training/test data | Free |

---

## 🧠 How It Works

### Traditional Models (Naive Bayes & Logistic Regression)
1. **Preprocessing** — lowercase, remove URLs/numbers/punctuation
2. **TF-IDF** — convert text to numbers (spam words get high scores)
3. **Training** — model learns which words predict spam vs ham
4. **Prediction** — applies learned patterns to new messages

### Zero-Shot Model (BART)
- No training needed — model already understands language
- We simply ask: "Is this SMS spam or ham?"
- Slower, but requires no labelled data

---

## ❓ Common Problems

**"FileNotFoundError: data/SMSSpamCollection"**
→ Make sure you downloaded the dataset and placed it in the `data/` folder

**"ModuleNotFoundError"**
→ Run: `pip install -r requirements.txt`

**"torch not found" during zero-shot step**
→ Run: `pip install torch` (or skip Cell 11 — it's optional)
