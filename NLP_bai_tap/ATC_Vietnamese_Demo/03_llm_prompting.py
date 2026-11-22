"""Zero-shot LLM prompting for UIT-VSFC using Hugging Face Inference API.

This script does NOT fine-tune the LLM. It sends one prompt per test example
through a hosted inference provider, so latency includes network/provider time.
"""
import os
import re
import time

from huggingface_hub import InferenceClient

from common import LABELS, RESULTS_DIR, load_split, get_eval_subset, evaluate_and_save, set_seed

# Llama 3 Instruct is the default. You can override this in the environment.
MODEL_ID = os.getenv("HF_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = "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"

SYSTEM_PROMPT = """Bạn là hệ thống phân loại cảm xúc câu tiếng Việt.
Hãy phân loại nội dung vào đúng MỘT trong ba nhãn:
- negative: cảm xúc tiêu cực, không hài lòng, phàn nàn.
- neutral: trung tính, chỉ nêu thông tin hoặc nhận xét không rõ tích cực/tiêu cực.
- positive: cảm xúc tích cực, hài lòng, khen ngợi.
Chỉ trả về đúng một từ: negative, neutral hoặc positive. Không giải thích, không thêm dấu câu."""


def parse_label(response: str) -> str:
    text = (response or "").strip().lower()
    # Accept a label even if the model accidentally adds a short explanation.
    for label in LABELS:
        if re.search(rf"\b{label}\b", text):
            return label
    return "unknown"


def classify_one(client: InferenceClient, text: str) -> tuple[str, str]:
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Văn bản cần phân loại:\n{text}"},
        ],
        max_tokens=12,
        temperature=0.0,
    )
    raw = response.choices[0].message.content or ""
    return parse_label(raw), raw


def main():
    set_seed()
    if not HF_TOKEN:
        raise RuntimeError(
            "Chưa có HF_TOKEN. Hãy tạo Hugging Face access token, chấp nhận điều khoản "
            "của model gated nếu cần, rồi đặt biến môi trường HF_TOKEN trước khi chạy."
        )

    test_df = get_eval_subset(load_split("test"))
    print(f"Model: {MODEL_ID}")
    print(f"Số mẫu gửi đến API: {len(test_df)}")
    print("Lưu ý: runtime bao gồm thời gian mạng và thời gian xử lý của provider.")

    client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)
    predictions = []
    raw_outputs = []
    failed = 0

    start = time.perf_counter()
    for index, text in enumerate(test_df["text"].tolist(), start=1):
        try:
            label, raw = classify_one(client, text)
        except Exception as exc:
            label, raw = "unknown", f"API_ERROR: {type(exc).__name__}: {exc}"
            failed += 1
            print(f"[{index}] API lỗi: {type(exc).__name__}: {exc}")

        if label == "unknown":
            failed += 1 if not raw.startswith("API_ERROR:") else 0
        predictions.append(label)
        raw_outputs.append(raw)

        if index % 10 == 0 or index == len(test_df):
            print(f"Đã xử lý {index}/{len(test_df)} mẫu")

    predict_time = time.perf_counter() - start

    result = evaluate_and_save(
        model_name="LLaMA-3 prompting (HF API)",
        texts=test_df["text"].tolist(),
        y_true=test_df["label"].tolist(),
        y_pred=predictions,
        train_time_sec=None,
        predict_time_sec=predict_time,
        extra={
            "model_id": MODEL_ID,
            "inference_mode": "zero-shot prompting via hosted API",
            "invalid_or_failed_outputs": failed,
            "note": "Prediction time includes network/provider latency; not directly comparable to local inference time.",
        },
    )

    # Save raw model responses for error analysis/audit.
    import pandas as pd
    out_path = RESULTS_DIR / "llm_raw_responses.csv"
    pd.DataFrame({
        "text": test_df["text"].tolist(),
        "gold_label": test_df["label"].tolist(),
        "predicted_label": predictions,
        "raw_response": raw_outputs,
    }).to_csv(out_path, index=False, encoding="utf-8-sig")
    print("Raw API responses saved to:", out_path)


if __name__ == "__main__":
    main()
