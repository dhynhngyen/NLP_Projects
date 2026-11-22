# Thực nghiệm Text-to-SQL với T5-small trên bộ dữ liệu Spider

Thư mục này triển khai thực nghiệm **Text-to-SQL**, chuyển câu hỏi bằng ngôn ngữ tự nhiên thành câu lệnh SQL. Mô hình sử dụng **T5-small** từ Hugging Face và được fine-tune trên bộ dữ liệu **Spider**. PyTorch thực hiện huấn luyện và suy luận; Transformers cung cấp mô hình và tokenizer T5; Gradio tạo giao diện demo trên trình duyệt.

> **Phạm vi ứng dụng:** Người dùng nhập câu hỏi và database schema, mô hình sinh câu lệnh SQL dự đoán. `app.py` chỉ sinh SQL, **không kết nối tới cơ sở dữ liệu và không thực thi câu SQL**.

## 1. Cấu trúc thư mục

Cấu trúc liên quan đến thực nghiệm:

```text
Thuc_nghiem_Text_to_SQL/
├── app.py                              # Giao diện Gradio và mã suy luận
├── Spider_data/
│   └── spider_data/
│       ├── database/
│       ├── test_database/
│       ├── train_spider.json
│       ├── train_others.json           # Có thể có tùy bản Spider
│       ├── dev.json
│       ├── tables.json
│       ├── dev_gold.sql
│       ├── test.json
│       ├── test_tables.json
│       ├── test_gold.sql
│       └── README.txt
└── outputs/
    └── t5_spider_text2sql/
        ├── checkpoint-1752/            # Một checkpoint trong quá trình fine-tune
        ├── checkpoint-2190/            # Checkpoint app.py hiện đang nạp
        └── final/                      # Bản model/tokenizer được lưu riêng sau fine-tune
```

Tên thư mục và chữ hoa/chữ thường phải khớp với đường dẫn trong mã nguồn, đặc biệt khi chạy trên Linux/Ubuntu.

### Thành phần cần thiết theo mục đích

- **Chạy giao diện demo:** cần `app.py` và thư mục model mà `MODEL_DIR` trong `app.py` trỏ tới. Với mã hiện tại, đó là `outputs/t5_spider_text2sql/checkpoint-2190/`.
- **Thực hiện lại fine-tuning:** cần mã nguồn huấn luyện, các tệp Spider được mã nguồn sử dụng (`train_spider.json`, `tables.json`, `dev.json`) và các thư viện huấn luyện.
- `outputs/t5_spider_text2sql/` lưu checkpoint và bản model được xuất sau fine-tuning. `app.py` hiện nạp trực tiếp `checkpoint-2190`, không tự động nạp thư mục `final`.

> **Lưu ý khi tải repository từ GitHub:** Nếu `.gitignore` loại trừ toàn bộ thư mục `outputs/`, model trong `checkpoint-2190/` sẽ không có trong repository sau khi clone. Khi đó, ứng dụng không thể chạy cho đến khi bạn tự tạo/tải model vào đúng đường dẫn, hoặc đổi `MODEL_DIR` sang một thư mục model được phân phối kèm dự án. Các file trọng số lớn nên được quản lý bằng Git LFS hoặc lưu ở nơi tải xuống riêng; không nên đưa toàn bộ checkpoint và optimizer state lên GitHub nếu không cần thiết.

## 2. Công nghệ và thư viện sử dụng

| Công nghệ/thư viện | Vai trò trong thực nghiệm |
| ------------------ | ------------------------- |
| Python | Ngôn ngữ triển khai mã nguồn |
| PyTorch | Nền tảng tensor, huấn luyện và suy luận mô hình |
| Transformers | Tải T5-small, tokenizer, `Seq2SeqTrainer` và sinh SQL |
| Datasets | Đưa dữ liệu Spider vào đối tượng `Dataset` và tiền xử lý theo batch |
| Accelerate | Hỗ trợ Transformers trong quá trình huấn luyện |
| SentencePiece | Thành phần tokenizer cần thiết cho T5 |
| Gradio | Tạo giao diện nhập câu hỏi/schema và hiển thị SQL |
| Safetensors | Đọc/lưu trọng số mô hình ở định dạng Safetensors khi được sử dụng |

### Phiên bản tham khảo của môi trường thực nghiệm

Các phiên bản dưới đây được ghi nhận trong môi trường dùng cho thực nghiệm:

| Thành phần | Phiên bản |
| ----------- | --------: |
| PyTorch | 2.14.1 (CUDA 12.6 nếu cài bản GPU) |
| Transformers | 5.22.1 |
| Datasets | 5.1.0 |
| Accelerate | 1.15.0 |
| SentencePiece | 0.2.2 |
| Gradio | 6.31.0 |
| Safetensors | 0.8.0 |

Nên sử dụng Python 3.11 trở lên và chọn phiên bản Python được hỗ trợ bởi bản PyTorch cài đặt. Máy không có GPU NVIDIA vẫn có thể chạy bằng CPU, nhưng fine-tuning và suy luận có thể chậm hơn.

## 3. Chuẩn bị môi trường Python

Mở PowerShell trên Windows hoặc Terminal trên Ubuntu/Linux. Di chuyển đến thư mục `Thuc_nghiem_Text_to_SQL` trước khi thực hiện các bước tiếp theo.

### Windows PowerShell

```powershell
cd duong-dan-den\NLP_tieu_luan\Thuc_nghiem_Text_to_SQL

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Ubuntu/Linux

```bash
cd /duong/dan/den/NLP_tieu_luan/Thuc_nghiem_Text_to_SQL

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Sau khi kích hoạt thành công, dấu nhắc Terminal thường hiển thị `(.venv)` ở đầu dòng. Các thư viện ở những bước sau sẽ được cài vào môi trường này.

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

Mã nguồn thực nghiệm không import `torchvision`, vì vậy chỉ cài `torch` là đủ cho các đoạn code đã cung cấp. Lệnh cài PyTorch phải phù hợp với hệ điều hành, phiên bản Python và nền tảng tính toán của máy. Có thể chọn cấu hình trên trang cài đặt chính thức: <https://pytorch.org/get-started/locally/>.

Với bản PyTorch dựng sẵn, thông thường không cần cài CUDA Toolkit chỉ để chạy mô hình; tuy nhiên máy dùng GPU vẫn cần driver NVIDIA tương thích.

### 4.2. Cài các thư viện cho fine-tuning và demo

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

```python
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
```

Nếu `CUDA available` trả về `True`, PyTorch đã nhận diện GPU. Nếu trả về `False`, chương trình vẫn có thể chạy bằng CPU; hãy kiểm tra lại driver NVIDIA và chọn đúng bản PyTorch nếu muốn sử dụng GPU.

Kiểm tra dependency sau khi cài:

```bash
python -m pip check
```

## 5. Chuẩn bị bộ dữ liệu Spider

Mã fine-tuning sử dụng các tệp trong `Spider_data/spider_data/`:

| Tệp/thư mục | Mục đích |
| ------------ | -------- |
| `train_spider.json` | Câu hỏi, database ID và câu SQL dùng để tạo dữ liệu huấn luyện |
| `tables.json` | Metadata schema: bảng, cột, khóa chính và khóa ngoại |
| `dev.json` | Dữ liệu validation dùng để đánh giá trong quá trình fine-tuning |
| `database/` | Các cơ sở dữ liệu SQLite của Spider |
| `dev_gold.sql`, `test.json`, `test_tables.json`, `test_gold.sql` | Các tệp phục vụ đánh giá/kiểm tra theo cấu trúc benchmark Spider; mã fine-tuning không đọc trực tiếp tất cả các tệp này |

Cấu trúc tối thiểu:

```text
Thuc_nghiem_Text_to_SQL/
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

Nên dùng đường dẫn tương đối có dấu `/` để thuận tiện cho cả Windows và Linux:

```python
with open('Spider_data/spider_data/train_spider.json', 'r', encoding='utf8') as f:
    q_records = json.load(f)

with open('Spider_data/spider_data/tables.json', 'r', encoding='utf8') as f:
    db_records = json.load(f)

with open('Spider_data/spider_data/dev.json', 'r', encoding='utf8') as f:
    dev_json_contents = json.load(f)
```

Hãy chạy mã fine-tuning từ thư mục `Thuc_nghiem_Text_to_SQL` để các đường dẫn tương đối ở trên được phân giải đúng. Trong `Seq2SeqTrainingArguments`, nên dùng `output_dir="outputs/t5_spider_text2sql"` để dùng cùng một kiểu đường dẫn trên Windows và Ubuntu/Linux.

## 6. Quy trình fine-tune T5-small

### 6.1. Tải mô hình và tokenizer T5-small

Mã nguồn sử dụng checkpoint gốc:

```python
model_name = "google-t5/t5-small"

tokenizer_t5 = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
```

Ở lần chạy đầu tiên, Transformers cần tải model và tokenizer từ Hugging Face Hub, do đó máy cần kết nối Internet. Sau khi tải xong, các tệp thường được lưu trong cache của Hugging Face.

### 6.2. Tạo dữ liệu đầu vào và đầu ra

Mỗi mẫu huấn luyện gồm:

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

| Tham số | Giá trị trong thực nghiệm | Ý nghĩa |
| ------- | ------------------------: | ------- |
| `num_train_epochs` | 5 | Huấn luyện tối đa 5 epoch |
| `per_device_train_batch_size` | 4 | 4 mẫu trên mỗi thiết bị trong một batch huấn luyện |
| `per_device_eval_batch_size` | 4 | 4 mẫu trên mỗi thiết bị trong một batch đánh giá |
| `gradient_accumulation_steps` | 4 | Tích lũy gradient qua 4 batch nhỏ trước khi cập nhật |
| `learning_rate` | `5e-5` | Tốc độ học ban đầu |
| `weight_decay` | `0.01` | Hệ số regularization weight decay |
| `warmup_steps` | 110 | Số bước warm-up learning rate |
| `eval_strategy` | `epoch` | Đánh giá sau mỗi epoch |
| `save_strategy` | `epoch` | Lưu checkpoint sau mỗi epoch |
| `logging_steps` | 50 | Ghi log sau mỗi 50 bước |
| `save_total_limit` | 2 | Giới hạn số checkpoint trung gian được giữ lại |
| `load_best_model_at_end` | `True` | Nạp checkpoint tốt nhất ở cuối quá trình train |
| `metric_for_best_model` | `eval_loss` | Chọn mô hình tốt nhất dựa trên validation loss |
| `fp16` | `False` | Không bật huấn luyện FP16 trong cấu hình này |
| `dataloader_num_workers` | 0 | Không dùng worker riêng cho DataLoader |

Thư mục lưu kết quả:

```text
outputs/t5_spider_text2sql/
```

Trong quá trình huấn luyện, Trainer tạo các thư mục checkpoint, ví dụ `checkpoint-1752/` và `checkpoint-2190/`. Mỗi checkpoint có thể chứa trọng số model, cấu hình, trạng thái Trainer và tokenizer/processing class nếu tokenizer được truyền cho Trainer. Việc lưu tokenizer cùng checkpoint giúp checkpoint có đủ thành phần cần thiết để nạp và tiếp tục sử dụng một cách nhất quán.

### 6.5. Lưu model và tokenizer sau fine-tuning

Sau khi huấn luyện hoàn tất, mã nguồn lưu model và tokenizer vào thư mục `final/` bằng hai lệnh:

```python
trainer.save_model(
    "./outputs/t5_spider_text2sql/final"
)

tokenizer_t5.save_pretrained(
    "./outputs/t5_spider_text2sql/final"
)
```

Hai lệnh có vai trò khác nhau:

- `trainer.save_model(...)` lưu model đã được Trainer giữ ở thời điểm gọi lệnh, cùng cấu hình cần thiết để nạp model. Nếu đã bật `load_best_model_at_end=True`, Trainer sẽ nạp checkpoint tốt nhất ở cuối quá trình huấn luyện; do đó model được lưu ở `final/` là model tốt nhất được nạp vào Trainer tại thời điểm đó.
- `tokenizer_t5.save_pretrained(...)` lưu tokenizer vào cùng thư mục `final/`, bao gồm các tệp cấu hình và dữ liệu tokenizer mà phiên bản tokenizer đang dùng yêu cầu.

**Tại sao tokenizer cũng xuất hiện trong các thư mục `checkpoint-*`?** Nếu tokenizer/processing class được truyền cho `Trainer` (ví dụ qua tham số `processing_class` ở các phiên bản Transformers mới), Trainer có thể tự lưu tokenizer cùng mỗi checkpoint khi lưu trạng thái huấn luyện. Vì thế, `checkpoint-1752/` và `checkpoint-2190/` có thể chứa các tệp như `tokenizer.json`, `tokenizer_config.json` hoặc các tệp tokenizer liên quan. Đây là hành vi nhằm giúp mỗi checkpoint có thể được nạp lại cùng bộ tokenizer tương ứng; không có nghĩa tokenizer chỉ được lưu ở checkpoint.

Với lệnh `tokenizer_t5.save_pretrained("./outputs/t5_spider_text2sql/final")`, tokenizer **cũng phải được lưu vào `final/` nếu lệnh đã chạy thành công**. Nếu không thấy các tệp tokenizer ở đó, hãy kiểm tra:

1. Cell/script chứa `tokenizer_t5.save_pretrained(...)` đã thực sự được chạy sau fine-tuning chưa.
2. Thư mục làm việc hiện tại lúc chạy là thư mục `Thuc_nghiem_Text_to_SQL` hay chưa. Đường dẫn `./outputs/...` được tính tương đối theo thư mục làm việc hiện tại.
3. Có đang kiểm tra đúng `outputs/t5_spider_text2sql/final/` được tạo bởi lần chạy đó không.

Có thể kiểm tra đường dẫn thực tế bằng:

```python
from pathlib import Path

final_dir = Path("./outputs/t5_spider_text2sql/final").resolve()
print(final_dir)
print([p.name for p in final_dir.iterdir()] if final_dir.exists() else "Chưa có thư mục final")
```

### 6.6. Chạy mã fine-tuning

Mở notebook hoặc file Python chứa đoạn mã fine-tuning của thực nghiệm và chạy từ đầu đến cuối, bảo đảm thư mục làm việc hiện tại là `Thuc_nghiem_Text_to_SQL`.

Nếu mã được lưu thành file `process.py`, chạy:

```bash
python process.py
```

Nếu mã nằm trong Jupyter Notebook, mở notebook bằng Jupyter và chạy các cell theo đúng thứ tự.

Sau khi hoàn tất, mã nguồn có thể in ra các thông tin như `Best checkpoint`, `Best validation loss` và `Global steps`. Model và tokenizer được lưu vào `outputs/t5_spider_text2sql/final/` theo hai lệnh ở mục 6.5.

## 7. Chạy giao diện Text-to-SQL bằng Gradio

Trong phiên bản `app.py` hiện tại, model được nạp từ checkpoint `checkpoint-2190`:

```python
MODEL_DIR = (
    Path(__file__).parent
    / "outputs"
    / "t5_spider_text2sql"
    / "checkpoint-2190"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
```

Điều này có nghĩa `app.py` được đặt trực tiếp trong thư mục `Thuc_nghiem_Text_to_SQL/`, cùng cấp với thư mục `outputs/`. Cấu trúc tương ứng:

```text
Thuc_nghiem_Text_to_SQL/
├── app.py
├── outputs/
│   └── t5_spider_text2sql/
│       ├── checkpoint-1752/
│       ├── checkpoint-2190/       # app.py hiện đang nạp checkpoint này
│       └── final/                 # Bản model/tokenizer được xuất riêng
└── Spider_data/
```

`Path(__file__).parent` xác định thư mục chứa chính file `app.py`. Vì vậy, đường dẫn model được xây dựng dựa trên vị trí file, không phụ thuộc Terminal đang mở ở thư mục nào.

**Phân biệt checkpoint và final:** `app.py` hiện đang nạp `checkpoint-2190`, không phải `final/`. Nếu muốn ứng dụng sử dụng model được xuất vào `final/`, hãy đổi phần cuối của `MODEL_DIR` từ `"checkpoint-2190"` thành `"final"`:

```python
MODEL_DIR = (
    Path(__file__).parent
    / "outputs"
    / "t5_spider_text2sql"
    / "final"
)
```

Thư mục được chọn phải có đầy đủ file model và tokenizer để `from_pretrained()` nạp thành công.

### Khởi chạy ứng dụng

Mở Terminal tại thư mục thực nghiệm:

```powershell
cd duong-dan-den\NLP_tieu_luan\Thuc_nghiem_Text_to_SQL
python app.py
```

Ứng dụng sẽ:

1. Chọn `cuda` nếu `torch.cuda.is_available()` trả về `True`; nếu không, chọn `cpu`.
2. Nạp tokenizer từ `MODEL_DIR`.
3. Nạp mô hình T5 đã fine-tune từ `MODEL_DIR`.
4. Chuyển mô hình sang thiết bị đã chọn và gọi `model.eval()`.
5. Khởi chạy giao diện Gradio.

Mở địa chỉ local mà Gradio in trong Terminal, thường là:

```text
http://127.0.0.1:7860
```

Để dừng ứng dụng, quay lại Terminal và nhấn `Ctrl + C`.

## 8. Cách sử dụng giao diện

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

Hàm `build_input()` trong `app.py` tạo định dạng:

```text
Question: <câu hỏi>
Schema: <database schema>
```

Vì vậy, schema nên nêu rõ tên bảng, tên cột và quan hệ khóa ngoại. Schema cần mô tả đúng cơ sở dữ liệu mà câu hỏi đề cập.

## 9. Cách `app.py` sinh SQL

Hàm `predict_sql(question, schema)` thực hiện các bước chính:

1. Kiểm tra câu hỏi và schema có bị để trống không.
2. Gọi `build_input()` để tạo chuỗi `Question: ...\nSchema: ...`.
3. Dùng tokenizer chuyển văn bản thành tensor token ID; cắt đầu vào ở tối đa 640 token.
4. Chuyển tensor sang cùng thiết bị với mô hình (GPU hoặc CPU).
5. Gọi `model.generate()` để sinh SQL với các tham số hiện tại:

   | Tham số | Giá trị | Ý nghĩa |
   | ------- | ------: | ------- |
   | `max_new_tokens` | 256 | Số token mới tối đa được sinh |
   | `num_beams` | 4 | Beam search giữ 4 nhánh ứng viên |
   | `do_sample` | `False` | Không lấy mẫu ngẫu nhiên |

6. Gọi `tokenizer.decode()` để chuyển token ID đầu ra thành chuỗi SQL.
7. Trả SQL cho Gradio hiển thị.

Ứng dụng dùng `torch.inference_mode()` khi suy luận để không tính gradient trong quá trình dự đoán.

## 10. Kiểm tra và xử lý lỗi thường gặp

### 10.1. Thiếu thư viện Python

Nếu gặp `ModuleNotFoundError`, hãy xác nhận đã kích hoạt đúng môi trường Python và cài các thư viện ở mục 4.2:

```bash
python -m pip install transformers datasets accelerate sentencepiece gradio safetensors
```

Nếu thiếu PyTorch, cài bản CPU hoặc GPU theo mục 4.1.

### 10.2. PyTorch không nhận GPU

Chạy:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Nếu `torch.cuda.is_available()` là `False`, ứng dụng vẫn chọn CPU. Muốn dùng GPU NVIDIA, kiểm tra driver và cài đúng wheel CUDA của PyTorch tại <https://pytorch.org/get-started/locally/>.

### 10.3. Không tìm thấy mô hình/tokenizer

Kiểm tra:

- `MODEL_DIR` trong `app.py` trỏ đúng thư mục tồn tại hay chưa.
- Thư mục đó có đầy đủ cấu hình, trọng số và tệp tokenizer không.
- Các file nằm trực tiếp trong thư mục checkpoint/model đã chọn, không bị lồng thêm một thư mục không mong muốn.
- Nếu dùng `checkpoint-2190`, thư mục `outputs/t5_spider_text2sql/checkpoint-2190/` phải tồn tại. Nếu dùng `final`, cần có `outputs/t5_spider_text2sql/final/`.

### 10.4. Không tìm thấy dữ liệu Spider khi fine-tune

Kiểm tra đường dẫn và chữ hoa/chữ thường của:

```text
Spider_data/spider_data/train_spider.json
Spider_data/spider_data/tables.json
Spider_data/spider_data/dev.json
```

Đồng thời bảo đảm đang chạy notebook/script từ thư mục `Thuc_nghiem_Text_to_SQL` hoặc điều chỉnh các đường dẫn tương đối cho phù hợp.

### 10.5. Lần đầu tải mô hình T5 bị lỗi kết nối

`AutoTokenizer.from_pretrained("google-t5/t5-small")` và `AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")` cần tải mô hình/tokenizer từ Hugging Face Hub nếu máy chưa có cache. Kiểm tra kết nối Internet rồi chạy lại.

### 10.6. SQL dự đoán không chính xác

Mô hình sinh SQL dựa trên câu hỏi và schema được cung cấp. Kết quả có thể sai bảng, sai cột, sai điều kiện JOIN hoặc sai logic. Cần kiểm tra schema đầu vào và xác thực SQL trước khi sử dụng. Ứng dụng demo hiện không thực thi câu SQL trên database.

## 11. Tài liệu tham khảo

- PyTorch – Get Started: <https://pytorch.org/get-started/locally/>
- PyTorch – Previous Versions: <https://pytorch.org/get-started/previous-versions/>
- Hugging Face Transformers: <https://huggingface.co/docs/transformers/>
- Hugging Face Datasets: <https://huggingface.co/docs/datasets/>
- Gradio – Interface: <https://www.gradio.app/docs/gradio/interface>
- Spider Benchmark: <https://yale-lily.github.io/spider>
- T5-small: <https://huggingface.co/google-t5/t5-small>

---

**Tóm tắt quy trình:** Tạo môi trường Python mới → cài PyTorch và các thư viện cần thiết → chuẩn bị Spider → fine-tune T5-small → lưu model và tokenizer vào `outputs/t5_spider_text2sql/final/` → chọn checkpoint/model muốn dùng trong `MODEL_DIR` của `app.py` (hiện tại là `checkpoint-2190`) → chạy `python app.py` → nhập câu hỏi và database schema để nhận SQL dự đoán.
