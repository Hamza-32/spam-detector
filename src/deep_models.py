import joblib
import numpy as np

from .config import (
    BATCH_SIZE,
    CNN_FILTERS,
    CNN_MODEL_PATH,
    DROPOUT,
    EMBED_DIM,
    EPOCHS,
    LSTM_MODEL_PATH,
    LSTM_UNITS,
    MAX_SEQUENCE_LEN,
    MAX_VOCAB_WORDS,
    TOKENIZER_PATH,
)


def _require_tf():
    try:
        from tensorflow.keras.layers import Conv1D
        return Conv1D
    except Exception as exc:
        raise ImportError(
            "TensorFlow is required for deep models. Install with: pip install -r requirements_dl.txt"
        ) from exc


def build_tokenizer(train_texts, num_words: int = MAX_VOCAB_WORDS):
    _require_tf()
    from tensorflow.keras.preprocessing.text import Tokenizer

    tokenizer = Tokenizer(num_words=num_words, oov_token="<unk>")
    tokenizer.fit_on_texts(train_texts)
    return tokenizer


def texts_to_padded_sequences(tokenizer, texts, max_len: int = MAX_SEQUENCE_LEN):
    _require_tf()
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")


def build_cnn_model(vocab_size: int):
    _require_tf()
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=EMBED_DIM, input_length=MAX_SEQUENCE_LEN),
        Conv1D(filters=CNN_FILTERS, kernel_size=3, activation="relu", padding="same"),
        GlobalMaxPooling1D(),
        Dropout(DROPOUT),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_lstm_model(vocab_size: int):
    _require_tf()
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=EMBED_DIM, input_length=MAX_SEQUENCE_LEN),
        LSTM(LSTM_UNITS),
        Dropout(DROPOUT),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_and_predict_deep_models(x_train, y_train, x_test, epochs: int = EPOCHS):
    _require_tf()

    tokenizer = build_tokenizer(x_train)
    vocab_size = min(MAX_VOCAB_WORDS, len(tokenizer.word_index) + 1)

    x_train_seq = texts_to_padded_sequences(tokenizer, x_train)
    x_test_seq = texts_to_padded_sequences(tokenizer, x_test)

    y_train_np = np.asarray(y_train).astype("float32")

    cnn_model = build_cnn_model(vocab_size=vocab_size)
    cnn_hist = cnn_model.fit(
        x_train_seq,
        y_train_np,
        validation_split=0.1,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        verbose=0,
    )
    cnn_probs = cnn_model.predict(x_test_seq, verbose=0).flatten()
    cnn_preds = (cnn_probs >= 0.5).astype(int)

    lstm_model = build_lstm_model(vocab_size=vocab_size)
    lstm_hist = lstm_model.fit(
        x_train_seq,
        y_train_np,
        validation_split=0.1,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        verbose=0,
    )
    lstm_probs = lstm_model.predict(x_test_seq, verbose=0).flatten()
    lstm_preds = (lstm_probs >= 0.5).astype(int)

    return {
        "tokenizer": tokenizer,
        "cnn_model": cnn_model,
        "lstm_model": lstm_model,
        "cnn_preds": cnn_preds,
        "lstm_preds": lstm_preds,
        "cnn_probs": cnn_probs,
        "lstm_probs": lstm_probs,
        "histories": {
            "cnn": cnn_hist.history,
            "lstm": lstm_hist.history,
        },
    }


def save_deep_artifacts(tokenizer, cnn_model, lstm_model):
    TOKENIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(tokenizer, TOKENIZER_PATH)
    cnn_model.save(CNN_MODEL_PATH)
    lstm_model.save(LSTM_MODEL_PATH)


def load_deep_artifacts():
    _require_tf()
    from tensorflow.keras.models import load_model

    tokenizer = joblib.load(TOKENIZER_PATH)
    cnn_model = load_model(CNN_MODEL_PATH)
    lstm_model = load_model(LSTM_MODEL_PATH)
    return tokenizer, cnn_model, lstm_model
