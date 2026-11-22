"""Common utilities shared by the three Vietnamese ATC experiments."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

# Label order used consistently by all three programs.
LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

# This default follows the mapping in the user's existing UIT-VSFC notebook.
# IMPORTANT: verify the original label files before using a different UIT-VSFC copy.
# A dataset wrapper may expose a different numeric ClassLabel ordering.
NUMERIC_LABEL_MAP = {
    "0": "negative",
    "1": "neutral",
    "2": "positive",
}

# Use the same fixed, stratified test subset for all three models in the demo.
# Set to None in common.py to evaluate the entire official test split.
EVAL_LIMIT = 100
RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def normalize_label(value) -> str:
    """Normalize integer/string labels to negative, neutral, or positive."""
    raw = str(value).strip().lower()
    if raw in NUMERIC_LABEL_MAP:
        return NUMERIC_LABEL_MAP[raw]
    # Handle CSV labels loaded as floats, e.g. 0.0 instead of 0.
    try:
        numeric = float(raw)
        if numeric.is_integer() and str(int(numeric)) in NUMERIC_LABEL_MAP:
            return NUMERIC_LABEL_MAP[str(int(numeric))]
    except (TypeError, ValueError):
        pass

    aliases = {
        "negative": "negative", "neg": "negative", "tiêu cực": "negative", "tieu cuc": "negative",
        "neutral": "neutral", "neu": "neutral", "trung tính": "neutral", "trung tinh": "neutral",
        "positive": "positive", "pos": "positive", "tích cực": "positive", "tich cuc": "positive",
    }
    if raw not in aliases:
        raise ValueError(
            f"Nhãn không hợp lệ: {value!r}. Hãy kiểm tra mapping trong common.py. "
            f"Các nhãn hợp lệ: {LABELS} hoặc số 0/1/2 theo NUMERIC_LABEL_MAP."
        )
    return aliases[raw]


def _resolve_column(df: pd.DataFrame, candidates: Iterable[str], kind: str) -> str:
    lower_to_original = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    raise ValueError(
        f"Không tìm thấy cột {kind}. Các cột hiện có: {list(df.columns)}. "
        f"Tên cột được hỗ trợ: {list(candidates)}"
    )


def load_split(split: str) -> pd.DataFrame:
    """Read data/{split}.csv; expected columns text/sentence and label/sentiment."""
    aliases = {"validation": ["validation.csv", "val.csv", "dev.csv"]}
    candidates = aliases.get(split, [f"{split}.csv"])
    csv_path = next((DATA_DIR / name for name in candidates if (DATA_DIR / name).exists()), None)
    if csv_path is None:
        raise FileNotFoundError(
            f"Không tìm thấy file cho split '{split}' trong {DATA_DIR}. "
            f"Hãy tạo {split}.csv (hoặc validation.csv/val.csv/dev.csv) với hai cột text và label. "
            "Xem README.md hoặc chạy prepare_uit_vsfc.py nếu bạn có các file TXT gốc."
        )

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    text_col = _resolve_column(df, ["text", "sentence", "content", "review"], "văn bản")
    label_col = _resolve_column(df, ["label", "sentiment", "labels"], "nhãn")

    output = pd.DataFrame({
        "text": df[text_col].fillna("").astype(str).str.strip(),
        "label": df[label_col].apply(normalize_label),
    })
    output = output[output["text"].str.len() > 0].reset_index(drop=True)
    if output.empty:
        raise ValueError(f"Split {split} không còn mẫu hợp lệ sau khi loại văn bản rỗng.")
    return output


def get_eval_subset(test_df: pd.DataFrame) -> pd.DataFrame:
    """Return a reproducible, stratified subset so the API LLM run is affordable."""
    if EVAL_LIMIT is None or len(test_df) <= EVAL_LIMIT:
        return test_df.reset_index(drop=True).copy()

    subset, _ = train_test_split(
        test_df,
        train_size=EVAL_LIMIT,
        random_state=RANDOM_SEED,
        stratify=test_df["label"],
    )
    return subset.reset_index(drop=True)


def segment_vietnamese(text: str) -> str:
    """Word-segment text for PhoBERT and TF-IDF/SVM."""
    from underthesea import word_tokenize
    return word_tokenize(str(text), format="text")


def evaluate_and_save(
    model_name: str,
    texts: list[str],
    y_true: list[str],
    y_pred: list[str],
    train_time_sec: float | None,
    predict_time_sec: float,
    extra: dict | None = None,
) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )

    result = {
        "model": model_name,
        "n_test": len(y_true),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "train_time_sec": train_time_sec,
        "predict_time_sec": float(predict_time_sec),
        "avg_predict_ms_per_sample": float(predict_time_sec * 1000 / max(len(y_true), 1)),
        "classification_report": report,
    }
    if extra:
        result.update(extra)

    safe_name = model_name.lower().replace(" ", "_").replace("+", "plus")
    with open(RESULTS_DIR / f"{safe_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    pd.DataFrame({
        "text": texts,
        "gold_label": y_true,
        "predicted_label": y_pred,
    }).to_csv(RESULTS_DIR / f"{safe_name}_predictions.csv", index=False, encoding="utf-8-sig")

    # Append a compact summary row for easy comparison.
    summary_path = RESULTS_DIR / "comparison.csv"
    summary_row = pd.DataFrame([{
        "model": model_name,
        "n_test": len(y_true),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "train_time_sec": train_time_sec,
        "predict_time_sec": float(predict_time_sec),
        "avg_predict_ms_per_sample": float(predict_time_sec * 1000 / max(len(y_true), 1)),
    }])
    if summary_path.exists():
        old = pd.read_csv(summary_path)
        old = old[old["model"] != model_name]
        summary_row = pd.concat([old, summary_row], ignore_index=True)
    summary_row.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n========== KẾT QUẢ ĐÁNH GIÁ ==========")
    print(f"Model: {model_name}")
    print(f"Test samples: {len(y_true)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    if train_time_sec is None:
        print("Training time: N/A (prompting, không fine-tune)")
    else:
        print(f"Training time: {train_time_sec:.2f} giây")
    print(f"Prediction time: {predict_time_sec:.2f} giây")
    print(f"Average: {predict_time_sec * 1000 / max(len(y_true), 1):.2f} ms/mẫu")
    print("Đã lưu kết quả vào:", RESULTS_DIR)
    return result
