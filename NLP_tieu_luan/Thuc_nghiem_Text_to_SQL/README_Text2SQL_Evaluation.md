# Đánh giá mô hình Text-to-SQL bằng EM và EX trên Spider

Tài liệu này hướng dẫn cách tái tạo bước đánh giá mô hình **T5-small đã fine-tune trên bộ dữ liệu Spider** bằng hai chỉ số:

- **EM (Exact Match / Exact Set Matching):** đánh giá mức độ khớp cấu trúc SQL dự đoán với SQL chuẩn theo bộ phân tích của Spider.
- **EX (Execution Accuracy):** thực thi SQL dự đoán và SQL chuẩn trên cùng cơ sở dữ liệu SQLite, sau đó so sánh kết quả.

Quy trình gồm: nạp mô hình đã fine-tune → sinh SQL dự đoán cho tập `dev` → tạo file SQL chuẩn đúng định dạng → chạy Spider evaluator để tính EM và EX.

> **Lưu ý:** Tài liệu này dùng tập `dev` để minh họa quy trình đánh giá. Khi báo cáo kết quả cuối cùng, cần phân biệt rõ kết quả trên `dev` và kết quả trên `test`. Không dùng tập train để báo cáo khả năng tổng quát hóa.

---

## 1. Cấu trúc thư mục

Đặt các thư mục và file theo cấu trúc tương tự dưới đây. Các đường dẫn trong script được tính tương đối từ thư mục `Thuc_nghiem_text_to_SQL`.

```text
Thuc_nghiem_text_to_SQL/
│
├── README.md
├── generate_dev_predictions.py
│
├── Spider_data/
│   └── spider_data/
│       ├── database/
│       │   ├── <db_id_1>/
│       │   │   └── <db_id_1>.sqlite
│       │   ├── <db_id_2>/
│       │   │   └── <db_id_2>.sqlite
│       │   └── ...
│       ├── dev.json
│       ├── dev_gold.sql          # Có thể có sẵn trong bộ dữ liệu
│       ├── tables.json
│       └── ...
│
├── Spider_evaluator/
│   ├── evaluation.py
│   ├── process_sql.py
│   └── ...
│
├── Text2SQL/
│   └── model_t5_spider/
│       ├── config.json
│       ├── spiece.model
│       ├── tokenizer_config.json
│       ├── model.safetensors hoặc pytorch_model.bin
│       └── ...
│
└── evaluation_outputs/
    ├── dev_predictions.sql
    └── dev_gold_for_eval.sql
```

### Các thành phần cần có

| Thành phần | Vai trò |
|---|---|
| `dev.json` | Chứa câu hỏi, SQL chuẩn (`query`) và mã database (`db_id`) của từng mẫu validation. |
| `tables.json` | Chứa schema: tên bảng, tên cột, khóa ngoại và thông tin liên quan. |
| `database/` | Chứa các database SQLite dùng để thực thi SQL khi tính EX. |
| `model_t5_spider/` | Thư mục mô hình T5 và tokenizer đã fine-tune. |
| `Spider_evaluator/` | Mã đánh giá chính thức của Spider, gồm `evaluation.py` và `process_sql.py`. |
| `generate_dev_predictions.py` | Script nạp mô hình, dựng input và sinh SQL dự đoán cho tập dev. |

**Quan trọng:** `model_t5_spider` phải chứa mô hình và tokenizer đã fine-tune. Trong thực nghiệm này, thư mục đó được lấy từ thư mục model đã lưu bằng `trainer.save_model(...)` và `tokenizer.save_pretrained(...)`. Nếu chưa có, hãy sao chép nội dung thư mục `outputs/t5_spider_text2sql/final` vào `Text2SQL/model_t5_spider`.

---

## 2. Chuẩn bị môi trường Python

Nên sử dụng một môi trường Python riêng cho thực nghiệm. Ví dụ dưới đây tạo môi trường mới tên `.venv_eval`.

### Windows PowerShell

Mở PowerShell tại thư mục `Thuc_nghiem_text_to_SQL`:

```powershell
python -m venv .venv_eval
.\.venv_eval\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Ubuntu / Linux

Mở Terminal tại thư mục `Thuc_nghiem_text_to_SQL`:

```bash
python3 -m venv .venv_eval
source .venv_eval/bin/activate
python -m pip install --upgrade pip
```

### Cài các thư viện cần thiết

Script sinh dự đoán cần PyTorch, Transformers và SentencePiece. Spider evaluator cần NLTK.

Trước tiên, cài **PyTorch** phù hợp với máy đang sử dụng:

- Máy có GPU NVIDIA: chọn lệnh cài đặt phù hợp với phiên bản driver/CUDA theo trang chính thức của PyTorch.
- Máy không có GPU NVIDIA: cài bản CPU.

Trang hướng dẫn: <https://pytorch.org/get-started/locally/>.

Sau đó cài các thư viện còn lại:

```bash
python -m pip install transformers sentencepiece nltk
```

Có thể kiểm tra môi trường:

```bash
python - <<'PY'
import torch
import transformers
import nltk

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Transformers:", transformers.__version__)
print("NLTK:", nltk.__version__)
PY
```

Nếu máy không có GPU hoặc PyTorch không nhận GPU, script vẫn có thể chạy bằng CPU; thời gian sinh dự đoán sẽ lâu hơn.

### Tải tài nguyên NLTK

Một số phiên bản NLTK cần dữ liệu tokenizer `punkt` và `punkt_tab` khi chạy bộ phân tích SQL của Spider. Chạy:

```bash
python -m nltk.downloader punkt punkt_tab
```

---

## 3. Lấy mã đánh giá Spider

Nếu repository của dự án chưa có thư mục `Spider_evaluator`, clone mã nguồn Spider chính thức tại thư mục `Thuc_nghiem_text_to_SQL`:

```bash
git clone --depth 1 https://github.com/taoyds/spider.git Spider_evaluator
```

Kiểm tra hai file quan trọng:

```text
Spider_evaluator/evaluation.py
Spider_evaluator/process_sql.py
```

Nếu thư mục `Spider_evaluator` đã có sẵn trong repository của dự án thì không cần clone lại.

Tài liệu đánh giá của Spider mô tả định dạng file đầu vào và lệnh chạy evaluator: <https://github.com/taoyds/spider#evaluation>.

---

## 4. Sinh SQL dự đoán cho tập dev

### 4.1. Tạo script

Tại thư mục gốc `Thuc_nghiem_text_to_SQL`, tạo file `generate_dev_predictions.py`. Có thể sử dụng nội dung script đi kèm repository của dự án.

Script thực hiện các công việc:

1. Đọc `dev.json` và `tables.json`.
2. Dựa vào `db_id` của từng câu hỏi để tìm schema tương ứng.
3. Tạo input theo định dạng đã dùng khi fine-tune:

   ```text
   Question: <câu hỏi>
   Schema: <mô tả schema>
   ```

4. Tokenize input và đưa tensor lên CPU hoặc GPU.
5. Dùng mô hình T5 đã fine-tune để sinh SQL.
6. Ghi SQL dự đoán và SQL chuẩn ra hai file riêng.

**Định dạng cần thống nhất:** khi fine-tune, validation và inference, input nên có dấu xuống dòng giữa `Question` và `Schema`. Ví dụ:

```python
input_text = (
    f"Question: {question}\n"
    f"Schema: {schema_text}"
)
```

### 4.2. Chạy script

Đảm bảo Terminal đang ở thư mục `Thuc_nghiem_text_to_SQL`, nơi có file `generate_dev_predictions.py`, rồi chạy:

```bash
python generate_dev_predictions.py
```

Nếu gặp lỗi CUDA out of memory, mở file `generate_dev_predictions.py` và giảm batch inference, ví dụ:

```python
BATCH_SIZE = 1
```

hoặc:

```python
BATCH_SIZE = 2
```

Chạy lại script sau khi thay đổi.

### 4.3. Hai file đầu ra

Sau khi chạy thành công, script tạo:

```text
evaluation_outputs/
├── dev_predictions.sql
└── dev_gold_for_eval.sql
```

Định dạng của hai file khác nhau:

**`dev_predictions.sql`:** mỗi dòng chỉ chứa một câu SQL do mô hình dự đoán.

```sql
SELECT count(*) FROM singer
SELECT name FROM singer WHERE birth_year > 1990
```

**`dev_gold_for_eval.sql`:** mỗi dòng gồm SQL chuẩn, một ký tự tab (`\t`) và `db_id`.

```text
SELECT count(*) FROM singer<TAB>singer
SELECT name FROM singer WHERE birth_year > 1990<TAB>singer
```

`<TAB>` ở ví dụ chỉ dùng để minh họa; trong file thật phải là ký tự tab, không phải các dấu cách. Đây là định dạng mà Spider evaluator yêu cầu.

### 4.4. Kiểm tra số lượng dòng

Số dự đoán phải bằng số mẫu trong `dev.json`, và số dòng trong file gold cũng phải tương ứng.

Windows PowerShell:

```powershell
(Get-Content evaluation_outputs/dev_predictions.sql).Count
(Get-Content evaluation_outputs/dev_gold_for_eval.sql).Count
```

Ubuntu / Linux:

```bash
wc -l evaluation_outputs/dev_predictions.sql evaluation_outputs/dev_gold_for_eval.sql
```

Nếu hai file có số dòng khác nhau, không tiếp tục đánh giá; hãy kiểm tra lại script sinh dự đoán.

---

## 5. Chạy Spider evaluator để tính EM và EX

Tại thư mục gốc `Thuc_nghiem_text_to_SQL`, chạy:

```bash
python Spider_evaluator/evaluation.py \
  --gold evaluation_outputs/dev_gold_for_eval.sql \
  --pred evaluation_outputs/dev_predictions.sql \
  --etype all \
  --db Spider_data/spider_data/database \
  --table Spider_data/spider_data/tables.json
```

Nếu dùng Windows PowerShell, có thể chạy cùng lệnh trên một dòng:

```powershell
python Spider_evaluator/evaluation.py --gold evaluation_outputs/dev_gold_for_eval.sql --pred evaluation_outputs/dev_predictions.sql --etype all --db Spider_data/spider_data/database --table Spider_data/spider_data/tables.json
```

### Ý nghĩa các tham số

| Tham số | Ý nghĩa |
|---|---|
| `--gold` | File SQL chuẩn; mỗi dòng có dạng `SQL<TAB>db_id`. |
| `--pred` | File SQL dự đoán; mỗi dòng chỉ có SQL. |
| `--etype all` | Chạy cả Exact Match và Execution Accuracy. |
| `--db` | Thư mục chứa các thư mục con của từng database SQLite. |
| `--table` | File `tables.json` chứa schema và quan hệ khóa ngoại. |

Có thể chạy riêng từng phép đánh giá:

**Chỉ EM / exact matching:**

```bash
python Spider_evaluator/evaluation.py \
  --gold evaluation_outputs/dev_gold_for_eval.sql \
  --pred evaluation_outputs/dev_predictions.sql \
  --etype match \
  --db Spider_data/spider_data/database \
  --table Spider_data/spider_data/tables.json
```

**Chỉ EX / execution:**

```bash
python Spider_evaluator/evaluation.py \
  --gold evaluation_outputs/dev_gold_for_eval.sql \
  --pred evaluation_outputs/dev_predictions.sql \
  --etype exec \
  --db Spider_data/spider_data/database \
  --table Spider_data/spider_data/tables.json
```

---

## 6. Đọc kết quả

Evaluator in kết quả theo mức độ khó của câu hỏi: `easy`, `medium`, `hard`, `extra` và `all`.

Khi báo cáo kết quả trên toàn bộ tập dev, chú ý cột `all`:

- **exact match – all:** tỷ lệ câu SQL dự đoán khớp theo tiêu chí cấu trúc của Spider.
- **execution – all:** tỷ lệ câu SQL dự đoán cho kết quả thực thi giống SQL chuẩn trên database tương ứng.

Ví dụ minh họa (không phải kết quả thực nghiệm):

```text
exact match - all: 0.250
execution   - all: 0.320
```

Có thể trình bày thành:

| Chỉ số | Điểm | Phần trăm |
|---|---:|---:|
| EM | 0.250 | 25.0% |
| EX | 0.320 | 32.0% |

Điểm số dạng `0.x` được nhân 100 để đổi sang phần trăm.

### Phân biệt EM và EX

- EM so sánh cấu trúc SQL theo bộ phân tích của Spider, không đơn giản chỉ so sánh chuỗi ký tự.
- EX chạy SQL dự đoán và SQL chuẩn trên database rồi so sánh kết quả. EX có thể có trường hợp dương tính giả: hai câu SQL khác nhau về ý nghĩa nhưng tình cờ trả về cùng kết quả trên database cụ thể.

Vì vậy, nên báo cáo cả EM và EX, đồng thời kiểm tra thủ công một số câu dự đoán tiêu biểu.

---

## 7. Xử lý lỗi UTF-8 khi tính EX (nếu gặp)

Một số database Spider có thể chứa giá trị văn bản không giải mã được theo UTF-8. Khi đó evaluator có thể dừng với lỗi tương tự:

```text
sqlite3.OperationalError:
Could not decode to UTF-8 column 'last_name' ...
```

Lỗi này xảy ra khi SQLite trả kết quả truy vấn và Python cố giải mã một giá trị `TEXT` không hợp lệ theo UTF-8. Nó không nhất thiết có nghĩa là mô hình sinh SQL lỗi.

### Cách xử lý cho bộ evaluator trong thực nghiệm này

Mở file:

```text
Spider_evaluator/evaluation.py
```

Tìm hàm:

```python
def eval_exec_match(db, p_str, g_str, pred, gold):
```

Trong hàm này, tìm đoạn:

```python
conn = sqlite3.connect(db)
cursor = conn.cursor()
```

Thay bằng:

```python
conn = sqlite3.connect(db)

# Đọc giá trị TEXT dưới dạng bytes để tránh lỗi giải mã UTF-8
# khi database có chuỗi chứa byte không hợp lệ.
conn.text_factory = bytes

cursor = conn.cursor()
```

Sau đó lưu file và chạy lại lệnh đánh giá `--etype all`.

**Giải thích:** `text_factory = bytes` yêu cầu sqlite3 trả các giá trị TEXT dưới dạng bytes, không cố giải mã tất cả thành UTF-8 ngay khi `fetchall()`. Vì evaluator so sánh kết quả truy vấn dự đoán và truy vấn chuẩn lấy từ cùng database, các giá trị text giống nhau sẽ được so sánh bằng bytes.

Đây là chỉnh sửa cục bộ để xử lý dữ liệu có byte không hợp lệ; nó không sửa database gốc. Không nên xem đây là cách khôi phục nội dung văn bản bị hỏng. Hãy ghi lại việc đã áp dụng chỉnh sửa này khi mô tả môi trường đánh giá để kết quả có thể tái lập.

---

## 8. Một số lỗi thường gặp

### 8.1. Không tìm thấy model hoặc tokenizer

Kiểm tra thư mục:

```text
Text2SQL/model_t5_spider/
```

Thư mục phải chứa cấu hình mô hình, trọng số và tokenizer đã fine-tune. Kiểm tra lại biến `MODEL_DIR` trong `generate_dev_predictions.py` nếu bạn đặt mô hình ở vị trí khác.

### 8.2. `FileNotFoundError` với `dev.json` hoặc `tables.json`

Đảm bảo cấu trúc dữ liệu đúng:

```text
Spider_data/spider_data/dev.json
Spider_data/spider_data/tables.json
```

Chạy script từ thư mục gốc `Thuc_nghiem_text_to_SQL`.

### 8.3. `CUDA out of memory`

Giảm `BATCH_SIZE` trong `generate_dev_predictions.py` xuống 1 hoặc 2. Có thể chạy bằng CPU nếu không có GPU tương thích.

### 8.4. Thiếu NLTK tokenizer data

Chạy lại:

```bash
python -m nltk.downloader punkt punkt_tab
```

### 8.5. Có nhiều dòng `eval_err_num`

`eval_err_num` là số câu SQL dự đoán không phân tích được theo parser của Spider. Đây không nhất thiết là lỗi làm evaluator dừng chương trình; các câu SQL không phân tích được sẽ bị tính là không khớp ở phần đánh giá. Nếu số lượng lớn, cần kiểm tra các câu trong `dev_predictions.sql`, cách tạo input, dữ liệu fine-tuning và chất lượng mô hình.

### 8.6. Không tìm thấy database tương ứng với `db_id`

Đảm bảo thư mục database có cấu trúc:

```text
Spider_data/spider_data/database/<db_id>/<db_id>.sqlite
```

Tên thư mục và tên file SQLite phải tương ứng với `db_id` trong `dev.json`.

---

## 9. Những gì cần lưu lại khi báo cáo thực nghiệm

Để kết quả có thể kiểm tra và tái lập, nên ghi lại:

- Tên mô hình nền: T5-small (`google-t5/t5-small`).
- Checkpoint được đánh giá: checkpoint tốt nhất hay checkpoint cuối cùng.
- Cách tạo input: `Question: ...\nSchema: ...`.
- Phiên bản thư viện PyTorch và Transformers.
- Phiên bản hoặc commit của Spider evaluator.
- Tập đánh giá: `dev` hay `test`.
- Kết quả EM và EX ở cột `all`.
- Có hay không chỉnh sửa `text_factory` để xử lý dữ liệu UTF-8.

**Lưu ý về phiên bản evaluator:** README này sử dụng evaluator gốc trong repository `taoyds/spider`. Nếu mã đánh giá được cập nhật trong tương lai, hãy ghi lại commit đang dùng vì các thay đổi ở evaluator có thể ảnh hưởng khả năng tái lập kết quả.

---

## 10. Tài liệu tham khảo

- Spider – repository chính thức: <https://github.com/taoyds/spider>
- Spider – hướng dẫn Evaluation: <https://github.com/taoyds/spider#evaluation>
- Spider – ví dụ file gold/prediction: <https://github.com/taoyds/spider/tree/master/evaluation_examples>
- PyTorch – hướng dẫn cài đặt: <https://pytorch.org/get-started/locally/>
- Hugging Face Transformers: <https://huggingface.co/docs/transformers/index>
