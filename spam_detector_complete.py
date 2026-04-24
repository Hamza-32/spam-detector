# ============================================================
# SMS SPAM DETECTOR - COMPLETE CODEBASE
# CSE-4218 Machine Learning Laboratory
# Bangladesh University of Professionals
# ============================================================
# HOW TO RUN:
#   1. Open terminal in this folder
#   2. Run: python spam_detector_complete.py
#   OR open as a Jupyter Notebook (copy each section into cells)
# ============================================================

# ── SECTION 1: IMPORTS ──────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import nltk
import re
import time
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    confusion_matrix, classification_report
)

# Download NLTK data (only needed once)
print("Downloading NLTK resources...")
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

print("=" * 55)
print("  SMS SPAM DETECTOR — Starting...")
print("=" * 55)


# ── SECTION 2: LOAD DATASET ─────────────────────────────────
print("\n[1/7] Loading dataset...")

df = pd.read_csv(
    'data/SMSSpamCollection',
    sep='\t',
    header=None,
    names=['label', 'message'],
    encoding='latin-1'
)

print(f"  Total messages loaded : {len(df)}")
print(f"  Ham  (legitimate)     : {(df['label']=='ham').sum()}")
print(f"  Spam                  : {(df['label']=='spam').sum()}")
print(f"  Spam percentage       : {(df['label']=='spam').mean()*100:.1f}%")

# Convert labels to numbers: ham=0, spam=1
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

print("\n  Sample data:")
print(df[['label', 'message']].head(3).to_string(index=False))


# ── SECTION 3: TEXT PREPROCESSING ───────────────────────────
print("\n[2/7] Preprocessing text...")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Cleans a raw SMS message:
    - Lowercase everything
    - Replace URLs with the token 'url'
    - Replace numbers with the token 'num'
    - Remove punctuation and special characters
    - Remove stop words (the, is, at, etc.)
    - Lemmatize (winning → win, calls → call)
    """
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', 'url', text)   # URLs
    text = re.sub(r'\d+', 'num', text)                # numbers
    text = re.sub(r'[^a-z\s]', '', text)              # punctuation
    tokens = text.split()
    tokens = [
        lemmatizer.lemmatize(w)
        for w in tokens
        if w not in stop_words and len(w) > 1
    ]
    return ' '.join(tokens)

df['clean_msg'] = df['message'].apply(clean_text)

print("  Before:", df['message'][1])
print("  After :", df['clean_msg'][1])
print("  Preprocessing complete!")


# ── SECTION 4: TRAIN/TEST SPLIT & FEATURE EXTRACTION ────────
print("\n[3/7] Splitting data and extracting features...")

X = df['clean_msg']
y = df['label_num']

# 80% training, 20% testing
# stratify=y ensures the same spam ratio in both splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"  Training set : {len(X_train)} messages")
print(f"  Test set     : {len(X_test)} messages")

# TF-IDF: converts text to numerical feature matrix
# max_features=5000 keeps only the top 5000 most useful words
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)   # includes single words AND word pairs
)

X_train_tfidf = tfidf.fit_transform(X_train)   # fit+transform on train
X_test_tfidf  = tfidf.transform(X_test)         # transform only on test

print(f"  Feature matrix shape : {X_train_tfidf.shape}")
print("  TF-IDF vectorisation done!")


# ── SECTION 5: TRAIN MODELS ─────────────────────────────────
print("\n[4/7] Training models...")

# --- Model 1: Multinomial Naive Bayes ---
print("\n  Training Naive Bayes...")
t0 = time.time()
nb_model = MultinomialNB(alpha=0.1)
nb_model.fit(X_train_tfidf, y_train)
nb_time = time.time() - t0
nb_preds = nb_model.predict(X_test_tfidf)
print(f"  Done in {nb_time*1000:.2f} ms")

# --- Model 2: Logistic Regression ---
print("\n  Training Logistic Regression...")
t0 = time.time()
lr_model = LogisticRegression(max_iter=1000, C=10, random_state=42)
lr_model.fit(X_train_tfidf, y_train)
lr_time = time.time() - t0
lr_preds = lr_model.predict(X_test_tfidf)
print(f"  Done in {lr_time*1000:.2f} ms")


# ── SECTION 6: EVALUATE MODELS ──────────────────────────────
print("\n[5/7] Evaluating models...")

def get_metrics(y_true, y_pred, name, train_time_ms):
    return {
        'Model'    : name,
        'Accuracy' : round(accuracy_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred), 4),
        'Recall'   : round(recall_score(y_true, y_pred), 4),
        'F1-Score' : round(f1_score(y_true, y_pred), 4),
        'Train (ms)': round(train_time_ms * 1000, 2)
    }

results = pd.DataFrame([
    get_metrics(y_test, nb_preds, 'Naive Bayes', nb_time),
    get_metrics(y_test, lr_preds, 'Logistic Regression', lr_time),
])

print("\n" + "="*65)
print(results.to_string(index=False))
print("="*65)

# Detailed classification reports
print("\n  Naive Bayes — Classification Report:")
print(classification_report(y_test, nb_preds, target_names=['Ham','Spam']))

print("\n  Logistic Regression — Classification Report:")
print(classification_report(y_test, lr_preds, target_names=['Ham','Spam']))


# ── SECTION 7: VISUALISATIONS ───────────────────────────────
print("\n[6/7] Generating charts...")

fig = plt.figure(figsize=(16, 12))
fig.suptitle('SMS Spam Detection — Results Dashboard',
             fontsize=16, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Plot 1: Class distribution
ax1 = fig.add_subplot(gs[0, 0])
counts = df['label'].value_counts()
colors = ['#378ADD', '#E24B4A']
bars = ax1.bar(counts.index, counts.values, color=colors, width=0.5, edgecolor='white')
ax1.set_title('Class Distribution', fontweight='bold')
ax1.set_ylabel('Count')
for bar, val in zip(bars, counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
             f'{val}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax1.set_ylim(0, max(counts.values) * 1.15)

# Plot 2: Confusion Matrix — Naive Bayes
ax2 = fig.add_subplot(gs[0, 1])
cm_nb = confusion_matrix(y_test, nb_preds)
sns.heatmap(cm_nb, annot=True, fmt='d', ax=ax2,
            cmap='Blues', linewidths=0.5,
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'],
            annot_kws={'size': 13, 'weight': 'bold'})
ax2.set_title('Naive Bayes\nConfusion Matrix', fontweight='bold')
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')

# Plot 3: Confusion Matrix — Logistic Regression
ax3 = fig.add_subplot(gs[0, 2])
cm_lr = confusion_matrix(y_test, lr_preds)
sns.heatmap(cm_lr, annot=True, fmt='d', ax=ax3,
            cmap='Reds', linewidths=0.5,
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'],
            annot_kws={'size': 13, 'weight': 'bold'})
ax3.set_title('Logistic Regression\nConfusion Matrix', fontweight='bold')
ax3.set_xlabel('Predicted')
ax3.set_ylabel('Actual')

# Plot 4: Metric Comparison Bar Chart
ax4 = fig.add_subplot(gs[1, :2])
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
nb_vals = [results[results['Model']=='Naive Bayes'][m].values[0] for m in metrics]
lr_vals = [results[results['Model']=='Logistic Regression'][m].values[0] for m in metrics]

x = np.arange(len(metrics))
w = 0.35
b1 = ax4.bar(x - w/2, nb_vals, w, label='Naive Bayes',
             color='#378ADD', edgecolor='white', linewidth=0.5)
b2 = ax4.bar(x + w/2, lr_vals, w, label='Logistic Regression',
             color='#E24B4A', edgecolor='white', linewidth=0.5)

for bar in list(b1) + list(b2):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

ax4.set_title('Model Performance Comparison', fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(metrics)
ax4.set_ylim(0.88, 1.02)
ax4.set_ylabel('Score')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

# Plot 5: Top spam words (Naive Bayes feature weights)
ax5 = fig.add_subplot(gs[1, 2])
feature_names = tfidf.get_feature_names_out()
spam_log_prob = nb_model.feature_log_prob_[1]
top_spam_idx  = spam_log_prob.argsort()[-12:][::-1]
top_words     = [feature_names[i] for i in top_spam_idx]
top_probs     = [spam_log_prob[i] for i in top_spam_idx]

ax5.barh(top_words[::-1], top_probs[::-1], color='#E24B4A', alpha=0.8)
ax5.set_title('Top Spam Keywords\n(Naive Bayes)', fontweight='bold')
ax5.set_xlabel('Log probability')
ax5.tick_params(axis='y', labelsize=9)

plt.savefig('results_dashboard.png', dpi=150, bbox_inches='tight')
print("  Chart saved as: results_dashboard.png")
plt.show()


# ── SECTION 8: ZERO-SHOT WITH FREE HUGGINGFACE MODEL ────────
print("\n[7/7] Zero-shot classification (free HuggingFace model)...")
print("  Note: First run downloads ~265 MB model. Subsequent runs are fast.")
print("  If you want to skip this, press Ctrl+C and comment out Section 8.")

try:
    from transformers import pipeline

    print("  Loading zero-shot classifier (facebook/bart-large-mnli)...")
    zs_classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1   # -1 = CPU (free), change to 0 if you have a GPU
    )

    candidate_labels = ["spam", "ham"]

    def zs_predict(text, classifier):
        result = classifier(
            text,
            candidate_labels=candidate_labels,
            hypothesis_template="This SMS message is {}."
        )
        # Return 1 (spam) if spam scored higher, else 0 (ham)
        return 1 if result['labels'][0] == 'spam' else 0

    # Use a small sample to save time (full test set would take hours on CPU)
    print("  Running on 100-message sample (full set takes ~30 min on CPU)...")
    SAMPLE_SIZE = 100
    X_test_list  = X_test.tolist()[:SAMPLE_SIZE]
    y_test_sample = y_test.tolist()[:SAMPLE_SIZE]

    t0 = time.time()
    zs_preds = [zs_predict(msg, zs_classifier) for msg in X_test_list]
    zs_time = time.time() - t0

    zs_metrics = get_metrics(y_test_sample, zs_preds,
                             'Zero-Shot (BART)', zs_time)
    print(f"\n  Zero-Shot Results (on {SAMPLE_SIZE} messages):")
    print(f"    Accuracy : {zs_metrics['Accuracy']}")
    print(f"    Precision: {zs_metrics['Precision']}")
    print(f"    Recall   : {zs_metrics['Recall']}")
    print(f"    F1-Score : {zs_metrics['F1-Score']}")
    print(f"    Time     : {zs_time:.1f}s")

    # Add to results for final table
    results = pd.concat([results, pd.DataFrame([zs_metrics])], ignore_index=True)

except KeyboardInterrupt:
    print("  Skipped zero-shot classification.")
except ImportError:
    print("  transformers not installed. Run: pip install transformers torch")
    print("  Skipping zero-shot section.")


# ── SECTION 9: TEST YOUR OWN MESSAGES ───────────────────────
print("\n" + "="*55)
print("  FINAL RESULTS TABLE")
print("="*55)
print(results[['Model','Accuracy','Precision','Recall','F1-Score']].to_string(index=False))

print("\n" + "="*55)
print("  TEST WITH YOUR OWN MESSAGES")
print("="*55)

my_messages = [
    "Congratulations! You've WON a FREE iPhone. Claim NOW: bit.ly/xyz",
    "Hey Hamza, are we still meeting tomorrow at 9?",
    "URGENT: Your bank account is locked. Verify at secure-bank.fakesite.com",
    "Can you send me yesterday's lecture slides?",
    "Win £1000 cash prize! Text WIN to 80085 NOW! Free entry!",
    "Don't forget the assignment submission is due Friday.",
    "You have been selected for a special offer. Call 09061123456",
]

print("\n  {:<48} {:>12} {:>12}".format("Message", "Naive Bayes", "Log.Regr."))
print("  " + "-"*74)

clean_my  = [clean_text(m) for m in my_messages]
my_tfidf  = tfidf.transform(clean_my)
nb_my     = nb_model.predict(my_tfidf)
lr_my     = lr_model.predict(my_tfidf)
label_map = {0: 'HAM ✓', 1: 'SPAM ✗'}

for msg, nb_p, lr_p in zip(my_messages, nb_my, lr_my):
    short = (msg[:46] + '..') if len(msg) > 48 else msg
    print(f"  {short:<48} {label_map[nb_p]:>12} {label_map[lr_p]:>12}")


# ── SECTION 10: SAVE MODELS ─────────────────────────────────
print("\n" + "="*55)
print("  SAVING MODELS")
print("="*55)
joblib.dump(nb_model, 'nb_model.pkl')
joblib.dump(lr_model, 'lr_model.pkl')
joblib.dump(tfidf,    'tfidf_vectorizer.pkl')
print("  Saved: nb_model.pkl")
print("  Saved: lr_model.pkl")
print("  Saved: tfidf_vectorizer.pkl")
print("  Saved: results_dashboard.png")
print("\n  All done! Your project is complete.")
