"""Fine-tune PhoBERT-base for 3-class UIT-VSFC sentiment classification."""
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from common import (
    LABELS, LABEL2ID, ID2LABEL, load_split, get_eval_subset,
    segment_vietnamese, evaluate_and_save, set_seed, ROOT_DIR,
)

MODEL_ID = "vinai/phobert-base"
MAX_LENGTH = 256


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        # PhoBERT requires Vietnamese text to be word-segmented before tokenization.
        segmented = [segment_vietnamese(text) for text in texts]
        self.encodings = tokenizer(
            segmented,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )
        self.labels = [LABEL2ID[label] for label in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: value[idx] for key, value in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
    }


def predict_in_batches(model, tokenizer, texts, batch_size, device):
    """Return predicted label strings and wall-clock inference time."""
    segmented = [segment_vietnamese(text) for text in texts]
    predictions = []
    model.eval()
    start = time.perf_counter()

    for i in range(0, len(segmented), batch_size):
        batch_texts = segmented[i:i + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits
        batch_preds = torch.argmax(logits, dim=-1).cpu().tolist()
        predictions.extend(ID2LABEL[int(label_id)] for label_id in batch_preds)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return predictions, elapsed


def main():
    set_seed()
    train_df = load_split("train")
    val_df = load_split("validation")
    test_df = get_eval_subset(load_split("test"))

    print(f"Train: {len(train_df)} | Validation: {len(val_df)} | Test đánh giá: {len(test_df)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Thiết bị:", device)
    if device.type == "cpu":
        print("CẢNH BÁO: PyTorch hiện không nhận CUDA; PhoBERT vẫn chạy được nhưng có thể chậm.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_dataset = SentimentDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer)
    val_dataset = SentimentDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer)
    test_texts = test_df["text"].tolist()

    output_dir = ROOT_DIR / "models" / "phobert_uit_vsfc"
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        report_to="none",
        fp16=False,
        dataloader_num_workers=0,
        seed=42,
        data_seed=42,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    print("Bắt đầu fine-tune PhoBERT...")
    train_start = time.perf_counter()
    trainer.train()
    train_time = time.perf_counter() - train_start

    # save_model stores the best checkpoint because load_best_model_at_end=True.
    trainer.save_model(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "best_model"))

    best_model = trainer.model.to(device)
    test_predictions, predict_time = predict_in_batches(
        best_model,
        tokenizer,
        test_texts,
        batch_size=16,
        device=device,
    )

    evaluate_and_save(
        model_name="PhoBERT-base fine-tuned",
        texts=test_texts,
        y_true=test_df["label"].tolist(),
        y_pred=test_predictions,
        train_time_sec=train_time,
        predict_time_sec=predict_time,
        extra={"base_model": MODEL_ID, "device": str(device), "epochs": 3},
    )


if __name__ == "__main__":
    main()
