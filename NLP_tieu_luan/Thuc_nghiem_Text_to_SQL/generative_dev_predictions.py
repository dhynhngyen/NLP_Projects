
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# =====================================================
# 1. KHAI BÁO ĐƯỜNG DẪN
# =====================================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "Spider_data" / "spider_data"

DEV_FILE = DATA_DIR / "dev.json"
TABLES_FILE = DATA_DIR / "tables.json"

# Mô hình đã fine-tune và tokenizer
MODEL_DIR = ROOT / "Text2SQL" / "model_t5_spider"

# Thư mục lưu kết quả đánh giá
OUTPUT_DIR = ROOT / "evaluation_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRED_FILE = OUTPUT_DIR / "dev_predictions.sql"
GOLD_FILE = OUTPUT_DIR / "dev_gold_for_eval.sql"

BATCH_SIZE = 4


# =====================================================
# 2. XÂY DỰNG DATABASE SCHEMA
# =====================================================

def build_schema_text(db_inf: dict) -> str:

    tables = db_inf["table_names_original"]
    columns = db_inf["column_names_original"]
    foreign_keys = db_inf["foreign_keys"]

    schema_lines = ["Tables info: "]

    # Lấy danh sách cột của từng bảng
    for table_idx, table_name in enumerate(tables):

        table_columns = [
            column_name
            for table_idx_in_columns, column_name in columns
            if table_idx_in_columns == table_idx
        ]

        schema_lines.append(
            f"{table_name}({', '.join(table_columns)})"
        )

    # Lấy quan hệ khóa ngoại
    relation_lines = []

    for col_idx1, col_idx2 in foreign_keys:

        table_idx1, col_name1 = columns[col_idx1]
        table_idx2, col_name2 = columns[col_idx2]

        if table_idx1 >= 0 and table_idx2 >= 0:

            relation_lines.append(
                f"{tables[table_idx1]}.{col_name1}"
                f"<-> {tables[table_idx2]}.{col_name2}"
            )

    if relation_lines:
        schema_lines.append("Column relations: ")
        schema_lines.extend(relation_lines)

    return "\n".join(schema_lines)


# =====================================================
# 3. ĐỌC DỮ LIỆU DEV VÀ TABLES.JSON
# =====================================================

with open(DEV_FILE, "r", encoding="utf-8") as f:
    dev_records = json.load(f)

with open(TABLES_FILE, "r", encoding="utf-8") as f:
    db_records = json.load(f)

# Tạo dictionary để tra cứu database theo db_id
db_mapping = {
    db["db_id"]: db
    for db in db_records
}

print("Number of dev examples:", len(dev_records))
print("Number of databases:", len(db_mapping))


# =====================================================
# 4. TẠO INPUT CHO MODEL VÀ GOLD SQL
# =====================================================

input_texts = []
gold_lines = []

for record in dev_records:

    db_id = record["db_id"]

    if db_id not in db_mapping:
        raise ValueError(f"Không tìm thấy database: {db_id}")

    db_inf = db_mapping[db_id]

    schema_text = build_schema_text(db_inf)

    # Phải thống nhất với format lúc fine-tune và app.py
    input_text = (
        f"Question: {record['question']}\n"
        f"Schema: {schema_text}"
    )

    input_texts.append(input_text)

    # Format gold mà Spider evaluator yêu cầu:
    # SQL \t db_id
    gold_sql = record["query"].replace("\t", " ").replace("\n", " ")

    gold_lines.append(
        f"{gold_sql}\t{db_id}"
    )


# =====================================================
# 5. LOAD MODEL VÀ TOKENIZER
# =====================================================

if not MODEL_DIR.exists():
    raise FileNotFoundError(
        f"Không tìm thấy thư mục model: {MODEL_DIR}"
    )

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

model.to(device)
model.eval()


# =====================================================
# 6. SINH SQL CHO TOÀN BỘ DEV DATASET
# =====================================================

all_predictions = []

with torch.inference_mode():

    for start in range(0, len(input_texts), BATCH_SIZE):

        batch_texts = input_texts[
            start:start + BATCH_SIZE
        ]

        # Tokenization theo batch
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=640
        )

        # Đưa tensor lên cùng thiết bị với model
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        # Sinh SQL
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=4,
            do_sample=False
        )

        # Giải mã token ID thành câu SQL
        batch_predictions = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )

        # Mỗi SQL phải nằm trên một dòng
        batch_predictions = [
            " ".join(sql.strip().split())
            for sql in batch_predictions
        ]

        all_predictions.extend(batch_predictions)

        print(
            f"Processed {min(start + BATCH_SIZE, len(input_texts))}"
            f"/{len(input_texts)} examples"
        )


# =====================================================
# 7. LƯU PREDICTIONS VÀ GOLD SQL
# =====================================================

if len(all_predictions) != len(gold_lines):
    raise RuntimeError(
        "Số lượng predictions không khớp với số lượng gold SQL!"
    )

with open(PRED_FILE, "w", encoding="utf-8") as f:
    for sql in all_predictions:
        f.write(sql + "\n")

with open(GOLD_FILE, "w", encoding="utf-8") as f:
    for line in gold_lines:
        f.write(line + "\n")


print("\n===== COMPLETED =====")
print("Predictions:", PRED_FILE)
print("Gold:", GOLD_FILE)
print("Total predictions:", len(all_predictions))

# Hiển thị thử 5 kết quả đầu tiên
for i in range(min(5, len(dev_records))):

    print(f"\nExample {i + 1}")
    print("Question:", dev_records[i]["question"])
    print("Gold SQL:", dev_records[i]["query"])
    print("Pred SQL:", all_predictions[i])
