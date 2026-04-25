import argparse

from src.config import MAX_SEQUENCE_LEN
from src.deep_models import load_deep_artifacts, texts_to_padded_sequences
from src.llm_classifier import predict_with_ollama
from src.preprocess import clean_text
from src.traditional_models import load_traditional_artifacts


def predict_message(message: str):
    cleaned = clean_text(message)
    nb, svm, tfidf = load_traditional_artifacts()

    vec = tfidf.transform([cleaned])
    nb_pred = int(nb.predict(vec)[0])
    svm_pred = int(svm.predict(vec)[0])

    out = {
        "Naive Bayes": nb_pred,
        "SVM": svm_pred,
    }

    try:
        tokenizer, cnn_model, lstm_model = load_deep_artifacts()
        seq = texts_to_padded_sequences(tokenizer, [cleaned], max_len=MAX_SEQUENCE_LEN)
        cnn_pred = int((cnn_model.predict(seq, verbose=0).flatten()[0] >= 0.5))
        lstm_pred = int((lstm_model.predict(seq, verbose=0).flatten()[0] >= 0.5))
        out["CNN"] = cnn_pred
        out["LSTM"] = lstm_pred
    except Exception as exc:
        out["CNN/LSTM"] = f"Unavailable ({exc})"

    try:
        llm_pred = int(predict_with_ollama([message])[0])
        out["LLM (Ollama)"] = llm_pred
    except Exception as exc:
        out["LLM (Ollama)"] = f"Unavailable ({exc})"

    return out


def label_name(value):
    if value in (0, 1):
        return "SPAM" if value == 1 else "HAM"
    return str(value)


def interactive_mode():
    print("=" * 58)
    print("SMS Spam Multi-Model Predictor")
    print("Type a message and press Enter. Type 'quit' to exit.")
    print("=" * 58)

    while True:
        message = input("\nEnter SMS: ").strip()
        if message.lower() in {"quit", "exit", "q"}:
            print("Bye.")
            break
        if not message:
            continue

        preds = predict_message(message)
        print("\nPredictions:")
        for model_name, pred in preds.items():
            print(f"- {model_name}: {label_name(pred)}")


def main():
    parser = argparse.ArgumentParser(description="Predict a message with all available SMS spam models")
    parser.add_argument("--message", type=str, default="", help="Optional one-shot message input")
    args = parser.parse_args()

    if args.message:
        preds = predict_message(args.message)
        for model_name, pred in preds.items():
            print(f"{model_name}: {label_name(pred)}")
        return

    interactive_mode()


if __name__ == "__main__":
    main()
