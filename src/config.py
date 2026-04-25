from pathlib import Path

SEED = 42
TEST_SIZE = 0.2

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

DATASET_PATH = DATA_DIR / "SMSSpamCollection"

# Traditional ML
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)

# Deep learning
MAX_VOCAB_WORDS = 10000
MAX_SEQUENCE_LEN = 40
EMBED_DIM = 128
CNN_FILTERS = 128
LSTM_UNITS = 64
DROPOUT = 0.3
BATCH_SIZE = 64
EPOCHS = 8

# Persisted artifacts
NB_MODEL_PATH = MODELS_DIR / "naive_bayes.pkl"
SVM_MODEL_PATH = MODELS_DIR / "svm.pkl"
CNN_MODEL_PATH = MODELS_DIR / "cnn_model.keras"
LSTM_MODEL_PATH = MODELS_DIR / "lstm_model.keras"
TOKENIZER_PATH = MODELS_DIR / "tokenizer.pkl"
TFIDF_PATH = MODELS_DIR / "tfidf.pkl"

# LLM
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
OPENAI_MODEL = "gpt-4o-mini"
