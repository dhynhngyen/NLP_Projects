# Thực nghiệm Text-to-SQL với T5-small trên bộ dữ liệu Spider

Thư mục này triển khai một thực nghiệm **Text-to-SQL**: chuyển câu hỏi được viết bằng ngôn ngữ tự nhiên thành câu lệnh SQL. Mô hình sử dụng **T5-small** từ Hugging Face, được fine-tune trên bộ dữ liệu **Spider**. PyTorch thực hiện quá trình huấn luyện và suy luận; thư viện Transformers cung cấp mô hình/tokenizer T5; Gradio tạo giao diện demo trên trình duyệt.

> **Phạm vi ứng dụng:** Người dùng nhập câu hỏi và database schema, mô hình sinh ra câu lệnh SQL dự đoán. `app.py` hiện chỉ sinh SQL, **không kết nối tới cơ sở dữ liệu và không thực thi câu SQL**.

## 1. Cấu trúc thư mục

Cấu trúc liên quan đến thực nghiệm:

```text
NLP_tieu_luan/
└── Thuc_nghiem_text_to_SQL/
    ├── outputs/
    │   └── t5_spider_text2sql/
    │       ├── checkpoint-.../          # Các checkpoint trong quá trình fine-tune
    │       └── final/                   # Mô hình và tokenizer được lưu sau fine-tune
    │
    ├── Spider_data/
    │   └── spider_data/
    │       ├── database/
    │       ├── test_database/
    │       ├── train_spider.json
    │       ├── train_others.json        # Có thể có tùy bản Spider
    │       ├── dev.json
    │       ├── tables.json
    │       ├── dev_gold.sql
    │       ├── test.json
    │       ├── test_tables.json
    │       ├── test_gold.sql
    │       └── README.txt
    │
    └── Text2SQL/
        ├── .gradio/                     # Dữ liệu/cache được Gradio tạo khi chạy (nếu có)
        ├── model_t5_spider/             # Mô hình T5 đã fine-tune và tokenizer dùng cho demo
        └── app.py                       # Mã nguồn giao diện Text-to-SQL
```

Tên thư mục và chữ hoa/chữ thường phải khớp với đường dẫn trong mã nguồn, đặc biệt khi chạy trên Linux/Ubuntu.

### Thành phần cần thiết theo mục đích

- **Chỉ chạy giao diện demo:** cần `Text2SQL/app.py` và mô hình đã fine-tune nằm trong `Text2SQL/model_t5_spider/`.
- **Thực hiện lại fine-tuning:** cần mã nguồn fine-tuning, các tệp Spider được mã nguồn sử dụng (`train_spider.json`, `tables.json`, `dev.json`) và các thư viện huấn luyện.
- `outputs/t5_spider_text2sql/` lưu checkpoint/kết quả fine-tuning. Thư mục này không được `app.py` đọc trực tiếp; ứng dụng nạp mô hình từ `Text2SQL/model_t5_spider/`.

## 2. Công nghệ và thư viện sử dụng

| Công nghệ/thư viện | Vai trò trong thực nghiệm                                           |
| ------------------ | ------------------------------------------------------------------- |
| Python             | Ngôn ngữ triển khai mã nguồn                                        |
| PyTorch            | Nền tảng tensor, huấn luyện và suy luận mô hình                     |
| Transformers       | Tải T5-small, tokenizer, `Seq2SeqTrainer` và sinh SQL               |
| Datasets           | Đưa dữ liệu Spider vào đối tượng `Dataset` và tiền xử lý theo batch |
| Accelerate         | Hỗ trợ Transformers trong quá trình huấn luyện                      |
| SentencePiece      | Thành phần tokenizer cần thiết cho T5                               |
| Gradio             | Tạo giao diện nhập câu hỏi/schema và hiển thị SQL                   |
| Safetensors        | Đọc/lưu trọng số mô hình ở định dạng Safetensors khi được sử dụng   |

### Phiên bản tham khảo của môi trường thực nghiệm

Các phiên bản dưới đây dựa trên môi trường đã dùng cho thực nghiệm:

| Thành phần    |                          Phiên bản |
| ------------- | ---------------------------------: |
| PyTorch       | 2.14.1 (CUDA 12.6 nếu cài bản GPU) |
| Transformers  |                             5.22.1 |
| Datasets      |                              5.1.0 |
| Accelerate    |                             1.15.0 |
| SentencePiece |                              0.2.2 |
| Gradio        |                             6.31.0 |
| Safetensors   |                              0.8.0 |

Nên sử dụng Python 3.11 trở lên và chọn phiên bản Python được hỗ trợ bởi bản PyTorch cài đặt. Máy không có GPU NVIDIA vẫn có thể chạy bằng CPU, nhưng fine-tuning và suy luận có thể chậm hơn.

## 3. Chuẩn bị môi trường Python

Mở PowerShell trên Windows hoặc Terminal trên Ubuntu/Linux. Di chuyển đến thư mục `Thuc_nghiem_text_to_SQL` trước khi thực hiện các bước tiếp theo.

### Windows PowerShell

```powershell
cd duong-dan-den\NLP_tieu_luan\Thuc_nghiem_text_to_SQL

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Ubuntu/Linux

```bash
cd /duong/dan/den/NLP_tieu_luan/Thuc_nghiem_text_to_SQL

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Sau khi kích hoạt thành công, dấu nhắc Terminal thường hiển thị `(.venv)` ở đầu dòng. Tất cả thư viện ở các bước sau sẽ được cài vào môi trường này.

## 4. Cài đặt các thư viện cần thiết

### 4.1. Cài PyTorch

Chọn **một** trong hai cách dưới đây.

**Máy có GPU NVIDIA và muốn chạy CUDA 12.6:**

```bash
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
```

**Chỉ chạy bằng CPU hoặc không có GPU NVIDIA:**

```bash
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
```

Mã nguồn thực nghiệm không import `torchvision`, vì vậy chỉ cài `torch` là đủ cho các đoạn code đã cung cấp. Lệnh cài PyTorch phải phù hợp với hệ điều hành, phiên bản Python và nền tảng tính toán của máy. Có thể chọn lại cấu hình trên trang cài đặt chính thức của PyTorch: <https://pytorch.org/get-started/locally/>.

Với bản PyTorch dựng sẵn, thông thường không cần cài CUDA Toolkit chỉ để chạy mô hình; tuy nhiên máy dùng GPU vẫn cần driver NVIDIA tương thích. Trên WSL2, driver NVIDIA được cài ở Windows và WSL2 truy cập GPU thông qua driver đó.

### 4.2. Cài các thư viện cho fine-tuning và demo

Sau khi cài PyTorch, chạy:

```bash
python -m pip install \
  transformers==5.22.1 \
  datasets==5.1.0 \
  accelerate==1.15.0 \
  sentencepiece==0.2.2 \
  gradio==6.31.0 \
  safetensors==0.8.0
```

Nếu mã fine-tuning được chạy trong Jupyter Notebook, cài thêm:

```bash
python -m pip install jupyter ipykernel
```

Các thư viện `json`, `pathlib` và một số mô-đun khác được sử dụng trong mã nguồn là thư viện chuẩn của Python, không cần cài riêng bằng pip.

### 4.3. Kiểm tra môi trường sau khi cài

```bash
python - <<'PY'
import sys
import torch
import transformers
import datasets
import accelerate
import sentencepiece
import gradio
import safetensors

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("Datasets:", datasets.__version__)
print("Accelerate:", accelerate.__version__)
print("SentencePiece:", sentencepiece.__version__)
print("Gradio:", gradio.__version__)
print("Safetensors:", safetensors.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Thiết bị: CPU")
PY
```

Nếu `CUDA available` trả về `True`, PyTorch đã nhận diện GPU. Nếu trả về `False`, chương trình vẫn có thể chạy bằng CPU; hãy kiểm tra lại driver NVIDIA và chọn đúng bản PyTorch nếu muốn sử dụng GPU.

Có thể kiểm tra các dependency sau khi cài:

```bash
python -m pip check
```

## 5. Chuẩn bị bộ dữ liệu Spider

Mã fine-tuning sử dụng các tệp sau trong `Spider_data/spider_data/`:

| Tệp                                                              | Mục đích                                                                                                                                   |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `train_spider.json`                                              | Các câu hỏi, database ID và câu SQL dùng để tạo dữ liệu huấn luyện                                                                         |
| `tables.json`                                                    | Thông tin schema: bảng, cột, khóa chính và khóa ngoại                                                                                      |
| `dev.json`                                                       | Dữ liệu validation dùng để đánh giá trong quá trình fine-tuning                                                                            |
| `database/`                                                      | Các cơ sở dữ liệu SQLite của Spider                                                                                                        |
| `dev_gold.sql`, `test.json`, `test_tables.json`, `test_gold.sql` | Các tệp phục vụ đánh giá/kiểm tra theo cấu trúc benchmark Spider; đoạn mã fine-tuning được cung cấp không đọc trực tiếp tất cả các tệp này |

Kiểm tra tối thiểu các đường dẫn:

```text
Thuc_nghiem_text_to_SQL/
└── Spider_data/
    └── spider_data/
        ├── train_spider.json
        ├── tables.json
        ├── dev.json
        ├── database/
        └── ...
```

Mã nguồn đọc `train_spider.json`, `tables.json` và `dev.json`, sau đó ghép mỗi câu hỏi với schema tương ứng thông qua trường `db_id`.

### Lưu ý về đường dẫn trong mã nguồn

Đoạn mã gốc có đường dẫn dạng Windows, ví dụ:

```python
with open('.\\Spider_data\\spider_data\\train_spider.json', 'r', encoding='utf8') as f:
```

Nếu chạy trên Ubuntu/Linux, nên đổi sang đường dẫn dùng dấu `/` để thuận tiện cho cả Windows và Linux:

```python
with open('Spider_data/spider_data/train_spider.json', 'r', encoding='utf8') as f:
    q_records = json.load(f)

with open('Spider_data/spider_data/tables.json', 'r', encoding='utf8') as f:
    db_records = json.load(f)

with open('Spider_data/spider_data/dev.json', 'r', encoding='utf8') as f:
    dev_json_contents = json.load(f)
```

Hãy chạy mã fine-tuning từ thư mục `Thuc_nghiem_text_to_SQL` để các đường dẫn tương đối ở trên được phân giải đúng. Trong `Seq2SeqTrainingArguments`, nếu `output_dir` đang được viết theo kiểu Windows như `".\\outputs\\t5_spider_text2sql"`, nên đổi thành `"outputs/t5_spider_text2sql"` để dùng cùng một đường dẫn trên Windows và Ubuntu/Linux.

## 6. Quy trình fine-tune T5-small

Phần này mô tả quy trình theo mã fine-tuning của thực nghiệm.

### 6.1. Tải mô hình và tokenizer T5-small

Mã nguồn sử dụng checkpoint gốc:

```python
model_name = "google-t5/t5-small"

tokenizer_t5 = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
```

Ở lần chạy đầu tiên, Transformers cần tải model và tokenizer từ Hugging Face Hub, do đó máy cần kết nối Internet. Sau khi tải xong, các tệp thường được lưu trong cache của Hugging Face.

### 6.2. Tạo dữ liệu đầu vào và đầu ra

Mỗi mẫu huấn luyện được tạo thành từ:

- **Input:** câu hỏi tự nhiên và database schema.
- **Output:** câu SQL tương ứng trong Spider.

Định dạng đầu vào:

```text
Question: <câu hỏi>
Schema: <thông tin bảng, cột và quan hệ>
```

Ví dụ:

```text
Question: How many singers are there?
Schema: Tables info:
singer(Singer_ID, Name, Birth_Year, Net_Worth_Millions, Citizenship)
song(Song_ID, Title, Singer_ID, Sales, Highest_Position)
Column relations:
song.Singer_ID <-> singer.Singer_ID
```

Hàm `build_schema_text()` lấy tên bảng, tên cột và quan hệ khóa ngoại từ metadata trong `tables.json`, rồi ghép thành chuỗi schema. Mã nguồn sử dụng `db_id` của từng câu hỏi để tìm đúng schema tương ứng.

### 6.3. Tokenize dữ liệu

Mã nguồn dùng tokenizer T5 để chuyển input và SQL mục tiêu thành token ID:

```python
model_inputs = tokenizer_t5(
    batch["input_text"],
    truncation=True,
    max_length=640
)

sql_query_encode = tokenizer_t5(
    batch["output"],
    truncation=True,
    max_length=256
)

model_inputs["labels"] = sql_query_encode["input_ids"]
```

- `max_length=640`: giới hạn độ dài phần input.
- `max_length=256`: giới hạn độ dài câu SQL mục tiêu.
- `labels`: chuỗi token ID của SQL chuẩn để mô hình học cách sinh đầu ra.

Sau đó, `Dataset.from_list()` tạo dataset và `.map(..., batched=True, batch_size=128)` tokenize dữ liệu theo từng nhóm 128 mẫu.

### 6.4. Cấu hình fine-tuning

Mã nguồn sử dụng `Seq2SeqTrainingArguments` với các thiết lập chính:

| Tham số                       | Giá trị trong thực nghiệm | Ý nghĩa                                                                   |
| ----------------------------- | ------------------------: | ------------------------------------------------------------------------- |
| `num_train_epochs`            |                         5 | Huấn luyện tối đa 5 epoch                                                 |
| `per_device_train_batch_size` |                         4 | 4 mẫu trên mỗi thiết bị trong một batch huấn luyện                        |
| `per_device_eval_batch_size`  |                         4 | 4 mẫu trên mỗi thiết bị trong một batch đánh giá                          |
| `gradient_accumulation_steps` |                         4 | Tích lũy gradient qua 4 batch nhỏ trước khi cập nhật theo batch hiệu dụng |
| `learning_rate`               |                    `5e-5` | Tốc độ học ban đầu                                                        |
| `weight_decay`                |                    `0.01` | Hệ số regularization weight decay                                         |
| `warmup_steps`                |                       110 | Số bước warm-up learning rate                                             |
| `eval_strategy`               |                   `epoch` | Đánh giá sau mỗi epoch                                                    |
| `save_strategy`               |                   `epoch` | Lưu checkpoint sau mỗi epoch                                              |
| `logging_steps`               |                        50 | Ghi log sau mỗi 50 bước                                                   |
| `save_total_limit`            |                         2 | Giới hạn số checkpoint được giữ lại                                       |
| `load_best_model_at_end`      |                    `True` | Nạp checkpoint tốt nhất ở cuối quá trình train                            |
| `metric_for_best_model`       |               `eval_loss` | Chọn mô hình tốt nhất dựa trên validation loss                            |
| `fp16`                        |                   `False` | Không bật huấn luyện FP16 trong cấu hình này                              |
| `dataloader_num_workers`      |                         0 | Không dùng worker riêng cho DataLoader                                    |

Đường dẫn lưu kết quả trong mã nguồn là:

```text
outputs/t5_spider_text2sql/
```

`save_total_limit=2` giới hạn số checkpoint trung gian được giữ lại. Sau khi huấn luyện, mã nguồn lưu mô hình và tokenizer vào:

```text
outputs/t5_spider_text2sql/final/
```

### 6.5. Chạy mã fine-tuning

Mở notebook hoặc file Python chứa đoạn mã fine-tuning của thực nghiệm và chạy từ đầu đến cuối, bảo đảm thư mục làm việc hiện tại là `Thuc_nghiem_text_to_SQL`.

Nếu bạn lưu mã thành một file Python có tên `process.py`, chạy:

```bash
python process.py
```

Nếu mã đang nằm trong Jupyter Notebook, mở notebook bằng Jupyter và chạy toàn bộ các cell theo đúng thứ tự.

Sau khi hoàn tất, mã nguồn in ra các thông tin như:

- `Best checkpoint`: checkpoint có validation loss tốt nhất.
- `Best validation loss`: giá trị validation loss tốt nhất.
- `Global steps`: tổng số bước cập nhật đã thực hiện.

Mã nguồn cũng lưu mô hình và tokenizer bằng `save_model()` và `save_pretrained()` vào thư mục `outputs/t5_spider_text2sql/final/`.

## 7. Chuẩn bị mô hình để chạy Gradio

Trong `app.py`, thư mục mô hình được xác định bằng:

```python
MODEL_DIR = Path(__file__).parent / "model_t5_spider"
```

Vì vậy, ứng dụng cần có cấu trúc:

```text
Text2SQL/
├── app.py
└── model_t5_spider/
    ├── config.json
    ├── model.safetensors              # hoặc pytorch_model.bin / các shard trọng số
    ├── tokenizer_config.json
    ├── spiece.model
    └── ...                            # các tệp cấu hình/tokenizer khác được lưu cùng mô hình
```

Tên tệp thực tế có thể thay đổi theo định dạng lưu mô hình. Cần có đầy đủ cấu hình, trọng số và tệp tokenizer mà `from_pretrained()` yêu cầu.

Sau khi fine-tune, mô hình nằm tại `outputs/t5_spider_text2sql/final/`. Hãy sao chép **nội dung bên trong** thư mục `final` sang `Text2SQL/model_t5_spider/`.

### Windows PowerShell

Chạy tại thư mục `Thuc_nghiem_text_to_SQL`:

```powershell
New-Item -ItemType Directory -Force .\Text2SQL\model_t5_spider
Copy-Item .\outputs\t5_spider_text2sql\final\* .\Text2SQL\model_t5_spider\ -Recurse -Force
```

### Ubuntu/Linux

Chạy tại thư mục `Thuc_nghiem_text_to_SQL`:

```bash
mkdir -p Text2SQL/model_t5_spider
cp -a outputs/t5_spider_text2sql/final/. Text2SQL/model_t5_spider/
```

Sau khi sao chép, các tệp mô hình phải nằm trực tiếp trong `model_t5_spider/`, không nằm sâu thêm trong `model_t5_spider/final/`.

## 8. Chạy giao diện Text-to-SQL bằng Gradio

Di chuyển vào thư mục chứa `app.py`:

```bash
cd Text2SQL
```

Khởi chạy:

```bash
python app.py
```

Ứng dụng thực hiện các bước:

1. Chọn `cuda` nếu `torch.cuda.is_available()` trả về `True`; nếu không, chọn `cpu`.
2. Nạp tokenizer từ `model_t5_spider/`.
3. Nạp mô hình T5 đã fine-tune từ `model_t5_spider/`.
4. Chuyển mô hình sang thiết bị đã chọn và gọi `model.eval()`.
5. Khởi chạy giao diện Gradio.

Mở địa chỉ local mà Gradio in trong Terminal, thường là:

```text
http://127.0.0.1:7860
```

Để dừng ứng dụng, quay lại Terminal và nhấn `Ctrl + C`.

## 9. Cách sử dụng giao diện

Giao diện có hai ô nhập:

1. **Câu hỏi bằng ngôn ngữ tự nhiên:** câu hỏi mà người dùng muốn chuyển thành SQL.
2. **Database Schema:** mô tả bảng, cột và quan hệ giữa các bảng của cơ sở dữ liệu.

Ví dụ câu hỏi:

```text
How many singers are there?
```

Ví dụ schema:

```text
Tables info:
singer(Singer_ID, Name, Birth_Year, Net_Worth_Millions, Citizenship)
song(Song_ID, Title, Singer_ID, Sales, Highest_Position)
Column relations:
song.Singer_ID <-> singer.Singer_ID
```

Nhấn nút gửi trong giao diện. SQL do mô hình dự đoán sẽ xuất hiện ở ô kết quả.

Định dạng input mà `app.py` tạo ra là:

```text
Question: <câu hỏi>
Schema: <database schema>
```

Vì vậy, schema nên nêu rõ tên bảng, tên cột và quan hệ khóa ngoại. Schema cần mô tả đúng cơ sở dữ liệu mà câu hỏi đề cập.

## 10. Cách `app.py` sinh SQL

Hàm `predict_sql(question, schema)` thực hiện các bước chính:

1. Kiểm tra câu hỏi và schema có bị để trống không.
2. Gọi `build_input()` để tạo chuỗi `Question: ...\nSchema: ...`.
3. Dùng tokenizer chuyển văn bản thành tensor token ID; cắt đầu vào ở tối đa 640 token.
4. Chuyển tensor sang cùng thiết bị với mô hình (GPU hoặc CPU).
5. Gọi `model.generate()` để sinh SQL với các tham số hiện tại:

   | Tham số          | Giá trị | Ý nghĩa                          |
   | ---------------- | ------: | -------------------------------- |
   | `max_new_tokens` |     256 | Số token mới tối đa được sinh    |
   | `num_beams`      |       4 | Beam search giữ 4 nhánh ứng viên |
   | `do_sample`      | `False` | Không lấy mẫu ngẫu nhiên         |

6. Gọi `tokenizer.decode()` để chuyển token ID đầu ra thành chuỗi SQL.
7. Trả SQL cho Gradio hiển thị.

## 11. Kiểm tra và xử lý lỗi thường gặp

### 11.1. Thiếu thư viện Python

Nếu gặp `ModuleNotFoundError`, hãy xác nhận đã kích hoạt đúng môi trường Python và cài các thư viện ở mục 4.2:

```bash
python -m pip install transformers datasets accelerate sentencepiece gradio safetensors
```

Nếu thiếu PyTorch, cài bản CPU hoặc GPU theo mục 4.1.

### 11.2. PyTorch không nhận GPU

Chạy:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Nếu `torch.cuda.is_available()` là `False`, ứng dụng vẫn chọn CPU. Muốn dùng GPU NVIDIA, kiểm tra driver và cài đúng wheel CUDA của PyTorch tại <https://pytorch.org/get-started/locally/>.

### 11.3. Không tìm thấy mô hình/tokenizer

Kiểm tra:

- Có thư mục `Text2SQL/model_t5_spider/` hay chưa.
- Thư mục có đầy đủ tệp cấu hình, trọng số và tokenizer không.
- Các tệp nằm trực tiếp trong `model_t5_spider/`, không bị lồng thêm một thư mục `final/`.

### 11.4. Không tìm thấy dữ liệu Spider khi fine-tune

Kiểm tra đường dẫn và chữ hoa/chữ thường của:

```text
Spider_data/spider_data/train_spider.json
Spider_data/spider_data/tables.json
Spider_data/spider_data/dev.json
```

Đồng thời bảo đảm đang chạy notebook/script từ thư mục `Thuc_nghiem_text_to_SQL` hoặc điều chỉnh các đường dẫn tương đối cho phù hợp.

### 11.5. Lần đầu tải mô hình T5 bị lỗi kết nối

`AutoTokenizer.from_pretrained("google-t5/t5-small")` và `AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")` cần tải mô hình/tokenizer từ Hugging Face Hub nếu máy chưa có cache. Kiểm tra kết nối Internet rồi chạy lại.

### 11.6. SQL dự đoán không chính xác

Mô hình sinh SQL dựa trên câu hỏi và schema được cung cấp. Kết quả có thể sai bảng, sai cột, sai điều kiện JOIN hoặc sai logic. Cần kiểm tra schema đầu vào và xác thực SQL trước khi sử dụng. Ứng dụng demo hiện không thực thi câu SQL trên database.

## 12. Tài liệu tham khảo

- PyTorch – Get Started: <https://pytorch.org/get-started/locally/>
- PyTorch – Previous Versions: <https://pytorch.org/get-started/previous-versions/>
- Hugging Face Transformers: <https://huggingface.co/docs/transformers/>
- Hugging Face Datasets: <https://huggingface.co/docs/datasets/>
- Gradio – Interface: <https://www.gradio.app/docs/gradio/interface>
- Spider Benchmark: <https://yale-lily.github.io/spider>
- T5-small: <https://huggingface.co/google-t5/t5-small>

---

**Tóm tắt quy trình:** Tạo môi trường Python mới → cài PyTorch và các thư viện cần thiết → chuẩn bị Spider → fine-tune T5-small → sao chép mô hình tốt nhất vào `Text2SQL/model_t5_spider/` → chạy `python app.py` → nhập câu hỏi và database schema để nhận SQL dự đoán.
