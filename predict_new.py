# ============================================================
# predict_new.py
# Use your saved model to classify any new message
# Run: python predict_new.py
# ============================================================
import joblib
import re
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
from nltk.corpus import stopwords
from nltk.stem  import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', 'url', text)
    text = re.sub(r'\d+', 'num', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens
              if w not in stop_words and len(w) > 1]
    return ' '.join(tokens)

# Load saved models
nb    = joblib.load('nb_model.pkl')
lr    = joblib.load('lr_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

print("=" * 50)
print("  SMS SPAM CLASSIFIER")
print("  Type a message and press Enter.")
print("  Type 'quit' to exit.")
print("=" * 50)

while True:
    msg = input("\n  Enter SMS: ").strip()
    if msg.lower() in ('quit', 'exit', 'q'):
        print("  Goodbye!")
        break
    if not msg:
        continue

    cleaned  = clean_text(msg)
    vec      = tfidf.transform([cleaned])
    nb_pred  = nb.predict(vec)[0]
    lr_pred  = lr.predict(vec)[0]

    nb_prob  = nb.predict_proba(vec)[0][nb_pred]
    lr_prob  = lr.predict_proba(vec)[0][lr_pred]

    label = {0: 'HAM  (legitimate)', 1: 'SPAM (unwanted)'}
    print(f"\n  Naive Bayes       → {label[nb_pred]}  ({nb_prob*100:.1f}% confident)")
    print(f"  Logistic Regr.    → {label[lr_pred]}  ({lr_prob*100:.1f}% confident)")
