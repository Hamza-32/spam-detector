import argparse
import time

from src.config import (
    DATASET_PATH,
    RESULTS_DIR,
    SEED,
)
from src.deep_models import save_deep_artifacts, train_and_predict_deep_models
from src.evaluate import compute_metrics, compute_roc, full_classification_report, make_confusion, results_table
from src.llm_classifier import predict_with_ollama
from src.preprocess import prepare_full_dataset, set_seed
from src.traditional_models import (
    predict,
    save_traditional_artifacts,
    train_naive_bayes,
    train_svm,
)


def run_pipeline(quick: bool = False, llm_sample: int = 100):
    from src.visualise import (
        plot_class_distribution,
        plot_confusion_matrices,
        plot_metric_bars,
        plot_roc_curves,
        plot_top_keywords,
        plot_training_curves,
    )

    set_seed(SEED)
    payload = prepare_full_dataset(DATASET_PATH)

    df = payload["df"]
    x_train = payload["x_train"]
    x_test = payload["x_test"]
    y_train = payload["y_train"]
    y_test = payload["y_test"]
    tfidf = payload["tfidf"]
    x_train_tfidf = payload["x_train_tfidf"]
    x_test_tfidf = payload["x_test_tfidf"]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    reports = {}
    confusions = {}
    roc_map = {}

    t0 = time.time()
    nb = train_naive_bayes(x_train_tfidf, y_train)
    nb_preds = predict(nb, x_test_tfidf)
    results.append(compute_metrics(y_test, nb_preds, "Naive Bayes"))
    reports["Naive Bayes"] = full_classification_report(y_test, nb_preds)
    confusions["Naive Bayes"] = make_confusion(y_test, nb_preds)
    roc_map["Naive Bayes"] = compute_roc(y_test, nb.predict_proba(x_test_tfidf)[:, 1])
    nb_time = time.time() - t0

    t0 = time.time()
    svm = train_svm(x_train_tfidf, y_train)
    svm_preds = predict(svm, x_test_tfidf)
    results.append(compute_metrics(y_test, svm_preds, "SVM"))
    reports["SVM"] = full_classification_report(y_test, svm_preds)
    confusions["SVM"] = make_confusion(y_test, svm_preds)
    roc_map["SVM"] = None
    svm_time = time.time() - t0

    deep_histories = {}
    try:
        t0 = time.time()
        deep = train_and_predict_deep_models(
            x_train,
            y_train,
            x_test,
            epochs=4 if quick else 8,
        )
        cnn_preds = deep["cnn_preds"]
        lstm_preds = deep["lstm_preds"]

        results.append(compute_metrics(y_test, cnn_preds, "CNN"))
        results.append(compute_metrics(y_test, lstm_preds, "LSTM"))
        reports["CNN"] = full_classification_report(y_test, cnn_preds)
        reports["LSTM"] = full_classification_report(y_test, lstm_preds)
        confusions["CNN"] = make_confusion(y_test, cnn_preds)
        confusions["LSTM"] = make_confusion(y_test, lstm_preds)
        roc_map["CNN"] = compute_roc(y_test, deep["cnn_probs"])
        roc_map["LSTM"] = compute_roc(y_test, deep["lstm_probs"])
        deep_histories = deep["histories"]
        deep_time = time.time() - t0

        save_deep_artifacts(deep["tokenizer"], deep["cnn_model"], deep["lstm_model"])
    except ImportError as exc:
        print(f"Deep models skipped: {exc}")
        deep_time = 0.0

    llm_count = min(llm_sample, len(x_test))
    llm_texts = x_test.iloc[:llm_count].tolist()
    llm_truth = y_test.iloc[:llm_count]
    try:
        t0 = time.time()
        llm_preds = predict_with_ollama(llm_texts)
        results.append(compute_metrics(llm_truth, llm_preds, "LLM (Ollama)"))
        reports["LLM (Ollama)"] = full_classification_report(llm_truth, llm_preds)
        confusions["LLM (Ollama)"] = make_confusion(llm_truth, llm_preds)
        roc_map["LLM (Ollama)"] = None
        llm_time = time.time() - t0
    except Exception as exc:
        print(f"LLM model skipped: {exc}")
        llm_time = 0.0

    save_traditional_artifacts(nb, svm, tfidf)

    table = results_table(results)
    table.to_csv(RESULTS_DIR / "model_comparison_results.csv", index=False)

    with open(RESULTS_DIR / "classification_reports.txt", "w", encoding="utf-8") as f:
        for model_name, report in reports.items():
            f.write("=" * 80 + "\n")
            f.write(model_name + "\n")
            f.write("=" * 80 + "\n")
            f.write(report + "\n")

    plot_confusion_matrices(confusions, RESULTS_DIR / "confusion_matrices.png")
    plot_metric_bars(table, RESULTS_DIR / "metric_bar_chart.png")
    plot_roc_curves(roc_map, RESULTS_DIR / "roc_curves.png")
    plot_top_keywords(tfidf, nb, RESULTS_DIR / "top_keywords_nb.png")
    plot_class_distribution(df, RESULTS_DIR / "class_distribution.png")
    if deep_histories:
        plot_training_curves(deep_histories, RESULTS_DIR / "training_curves.png")

    print("\nModel comparison:")
    print(table.to_string(index=False))
    print("\nSaved chart files in results/")
    print(f"Timing summary (seconds): NB={nb_time:.2f}, SVM={svm_time:.2f}, DEEP={deep_time:.2f}, LLM={llm_time:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Run full SMS spam comparison pipeline")
    parser.add_argument("--quick", action="store_true", help="Use fewer epochs and smaller LLM evaluation sample")
    parser.add_argument("--llm-sample", type=int, default=100, help="How many test rows to evaluate with LLM")
    args = parser.parse_args()

    run_pipeline(quick=args.quick, llm_sample=args.llm_sample)


if __name__ == "__main__":
    main()
