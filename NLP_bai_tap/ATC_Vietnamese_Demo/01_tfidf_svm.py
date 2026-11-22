"""Baseline: Vietnamese word segmentation + TF-IDF + Linear SVM."""
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from common import load_split, get_eval_subset, segment_vietnamese, evaluate_and_save, set_seed


def main():
    set_seed()
    train_df = load_split("train")
    test_df = get_eval_subset(load_split("test"))

    print(f"Train: {len(train_df)} | Test đánh giá: {len(test_df)}")
    print("Đang tách từ tiếng Việt và huấn luyện TF-IDF + SVM...")

    train_start = time.perf_counter()
    train_texts = [segment_vietnamese(t) for t in train_df["text"].tolist()]
    test_texts_segmented = [segment_vietnamese(t) for t in test_df["text"].tolist()]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=100_000,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_texts)
    classifier = LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=5000)
    classifier.fit(X_train, train_df["label"].tolist())
    train_time = time.perf_counter() - train_start

    predict_start = time.perf_counter()
    X_test = vectorizer.transform(test_texts_segmented)
    predictions = classifier.predict(X_test).tolist()
    predict_time = time.perf_counter() - predict_start

    evaluate_and_save(
        model_name="TF-IDF + Linear SVM",
        texts=test_df["text"].tolist(),
        y_true=test_df["label"].tolist(),
        y_pred=predictions,
        train_time_sec=train_time,
        predict_time_sec=predict_time,
    )


if __name__ == "__main__":
    main()
