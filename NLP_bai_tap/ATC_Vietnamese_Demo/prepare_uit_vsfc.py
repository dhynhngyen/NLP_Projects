"""Convert parallel UIT-VSFC TXT files into train/validation/test CSV files.

Expected raw files (put them under raw_uit_vsfc/):
  raw_uit_vsfc/train/sentences.txt
  raw_uit_vsfc/train/sentiments.txt
  raw_uit_vsfc/validation/sentences.txt
  raw_uit_vsfc/validation/sentiments.txt
  raw_uit_vsfc/test/sentences.txt
  raw_uit_vsfc/test/sentiments.txt

Also accepts files such as train_sentences.txt and train_sentiments.txt.
"""
from pathlib import Path
import pandas as pd

from common import ROOT_DIR, DATA_DIR, normalize_label

RAW_DIR = ROOT_DIR / "raw_uit_vsfc"


def find_pair(split: str):
    aliases = [split]
    if split == "validation":
        aliases += ["val", "dev"]

    sentence_names = ["sentences.txt", "sentence.txt", "sents.txt"]
    label_names = ["sentiments.txt", "sentiment.txt", "labels.txt"]

    for alias in aliases:
        for s_name in sentence_names:
            for l_name in label_names:
                s_path = RAW_DIR / alias / s_name
                l_path = RAW_DIR / alias / l_name
                if s_path.exists() and l_path.exists():
                    return s_path, l_path
        for s_name in [f"{alias}_sentences.txt", f"{alias}_sentence.txt"]:
            for l_name in [f"{alias}_sentiments.txt", f"{alias}_sentiment.txt", f"{alias}_labels.txt"]:
                s_path = RAW_DIR / s_name
                l_path = RAW_DIR / l_name
                if s_path.exists() and l_path.exists():
                    return s_path, l_path
    raise FileNotFoundError(
        f"Không tìm thấy cặp file sentence/label cho split {split} trong {RAW_DIR}. "
        "Xem cấu trúc mẫu trong README.md."
    )


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split in ["train", "validation", "test"]:
        sentence_path, label_path = find_pair(split)
        texts = sentence_path.read_text(encoding="utf-8").splitlines()
        raw_labels = label_path.read_text(encoding="utf-8").splitlines()
        if len(texts) != len(raw_labels):
            raise ValueError(
                f"{split}: số câu ({len(texts)}) khác số nhãn ({len(raw_labels)})."
            )

        df = pd.DataFrame({
            "text": [x.strip() for x in texts],
            "label": [normalize_label(x) for x in raw_labels],
        })
        df = df[df["text"].str.len() > 0].reset_index(drop=True)
        out_path = DATA_DIR / f"{split}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"{split}: {len(df)} mẫu -> {out_path}")
        print(df["label"].value_counts().to_dict())


if __name__ == "__main__":
    main()
