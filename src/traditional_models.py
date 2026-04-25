import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from .config import NB_MODEL_PATH, SVM_MODEL_PATH, TFIDF_PATH


def train_naive_bayes(x_train_tfidf, y_train, alpha: float = 0.1):
    model = MultinomialNB(alpha=alpha)
    model.fit(x_train_tfidf, y_train)
    return model


def train_svm(x_train_tfidf, y_train):
    model = LinearSVC()
    model.fit(x_train_tfidf, y_train)
    return model


def predict(model, x_tfidf):
    return model.predict(x_tfidf)


def save_traditional_artifacts(nb_model, svm_model, tfidf):
    NB_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(nb_model, NB_MODEL_PATH)
    joblib.dump(svm_model, SVM_MODEL_PATH)
    joblib.dump(tfidf, TFIDF_PATH)


def load_traditional_artifacts():
    nb_model = joblib.load(NB_MODEL_PATH)
    svm_model = joblib.load(SVM_MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)
    return nb_model, svm_model, tfidf
