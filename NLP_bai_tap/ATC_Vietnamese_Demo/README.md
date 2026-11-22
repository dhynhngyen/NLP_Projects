# Demo so sánh 3 nhóm mô hình ATC tiếng Việt trên UIT-VSFC

Bộ mã này triển khai cùng một tác vụ: **phân loại cảm xúc 3 lớp** của UIT-VSFC:

- `negative`: tiêu cực
- `neutral`: trung tính
- `positive`: tích cực

Ba phương pháp:
1. `01_tfidf_svm.py`: tách từ tiếng Việt → TF-IDF → Linear SVM.
2. `02_phobert_finetune.py`: tách từ tiếng Việt → fine-tune `vinai/phobert-base`.
3. `03_llm_prompting.py`: zero-shot prompting với LLaMA 3 qua Hugging Face Inference API (không fine-tune LLM).

## 1. Cấu trúc thư mục dữ liệu

Tạo ba file CSV:

```text
data/
├── train.csv
├── validation.csv
└── test.csv
```

Mỗi file cần có hai cột:

```csv
text,label
"Môn học rất hữu ích và dễ hiểu",positive
"Tôi không hài lòng với cách tổ chức môn học",negative
"Môn học được tổ chức vào thứ Hai",neutral
```

`load_split()` cũng chấp nhận tên cột văn bản là `sentence`, `content`, `review`; tên cột nhãn là `sentiment` hoặc `labels`.

### Nếu bạn đang có UIT-VSFC dạng các file TXT song song

Bạn có thể đặt file theo cấu trúc:

```text
raw_uit_vsfc/
├── train/
│   ├── sentences.txt
│   └── sentiments.txt
├── validation/
│   ├── sentences.txt
│   └── sentiments.txt
└── test/
    ├── sentences.txt
    └── sentiments.txt
```

Mỗi dòng trong `sentences.txt` tương ứng với nhãn ở cùng dòng trong `sentiments.txt`. Sau đó chạy:

```bash
python prepare_uit_vsfc.py
```

Script sẽ tạo `data/train.csv`, `data/validation.csv`, `data/test.csv`.

**Rất quan trọng về mapping nhãn số:** `common.py` hiện giả định đúng mapping theo notebook cũ của bạn: `0 = negative`, `1 = neutral`, `2 = positive`. Nếu file gốc của bạn dùng thứ tự khác, hãy sửa `NUMERIC_LABEL_MAP` trong `common.py` trước khi chạy. Một số bộ chuyển đổi/dataset wrapper có thể dùng thứ tự ClassLabel khác, nên hãy kiểm tra nhãn và vài mẫu thực tế trước khi đánh giá.

## 2. Cài đặt

Khuyến nghị tạo môi trường ảo riêng. Cài thư viện:

```bash
python -m pip install -r requirements.txt
```

PyTorch có thể cần cài riêng theo cấu hình CPU/CUDA trên máy của bạn. Nếu máy đang dùng bản PyTorch CPU-only thì PhoBERT vẫn chạy bằng CPU, nhưng chậm hơn. Không bắt buộc phải có GPU để chạy SVM hoặc LLM prompting qua API.

## 3. Chạy ba chương trình

Chạy từng file từ thư mục `ATC_Vietnamese_Demo`:

```bash
python 01_tfidf_svm.py
python 02_phobert_finetune.py
python 03_llm_prompting.py
```

PhoBERT tải model `vinai/phobert-base`, fine-tune 3 epochs, chọn checkpoint tốt nhất theo Macro-F1 trên validation và lưu model vào `models/phobert_uit_vsfc/best_model`.

## 4. LLM prompting bằng Hugging Face

Script LLM mặc định dùng `meta-llama/Llama-3.1-8B-Instruct` thông qua Hugging Face Inference API. Bạn cần:

1. Có Hugging Face account và access token có quyền gọi Inference Providers.
2. Chấp nhận giấy phép/điều kiện truy cập model LLaMA nếu tài khoản được yêu cầu.
3. Kiểm tra model có provider phục vụ tại thời điểm chạy; availability có thể thay đổi.

Trên PowerShell, đặt token trong phiên Terminal hiện tại (không ghi token vào source code):

```powershell
$env:HF_TOKEN="hf_your_token_here"
python 03_llm_prompting.py
```

Có thể đổi model ID bằng biến môi trường:

```powershell
$env:HF_MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
```

Bạn có thể thay bằng một model LLaMA 3.1 Instruct khác được provider hỗ trợ. Vistral-7B có thể chạy local qua Transformers hoặc GGUF/llama.cpp, nhưng không nên giả định model đó luôn có sẵn trên Hugging Face hosted provider. Model card Vistral: https://huggingface.co/finn10796/vistral-7b.

## 5. Cách đo và so sánh

- Accuracy và Macro-F1 được tính trên cùng một tập con test có phân tầng.
- Mặc định `EVAL_LIMIT = 100` trong `common.py`, để không gửi hàng nghìn request đến LLM API. Cả SVM và PhoBERT cũng dùng đúng 100 mẫu test đó để so sánh công bằng.
- Để đánh giá toàn bộ test set, đổi `EVAL_LIMIT = None`. Lưu ý LLM API có thể tốn thời gian/chi phí hoặc gặp rate limit nếu gửi nhiều request.
- SVM training time gồm tách từ, fit TF-IDF và fit Linear SVM.
- PhoBERT training time đo thời gian `trainer.train()`; không gồm thời gian tải model/tokenizer ban đầu.
- LLM không fine-tune nên training time là N/A. Prediction time của LLM là thời gian wall-clock qua mạng, gồm cả latency của provider; không nên diễn giải nó như thời gian tính toán thuần của model trên cùng phần cứng.

Kết quả lưu trong `results/`:

```text
results/
├── comparison.csv
├── tf-idf_plus_linear_svm_metrics.json
├── tf-idf_plus_linear_svm_predictions.csv
├── phobert-base_fine-tuned_metrics.json
├── phobert-base_fine-tuned_predictions.csv
├── llama-3_prompting_(hf_api)_metrics.json
├── llama-3_prompting_(hf_api)_predictions.csv
└── llm_raw_responses.csv
```

Tên file thực tế được chuẩn hóa từ tên model trong script.

## 6. Ghi chú phương pháp luận

- Dùng đúng official train/validation/test split, không gộp test vào train.
- Cả ba chương trình dự đoán cùng một nhãn cảm xúc 3 lớp. UIT-VSFC cũng có nhãn chủ đề 4 lớp; bộ code này không đánh giá topic classification.
- Đây là demo baseline để có số liệu ban đầu. Để viết báo cáo/nghiên cứu, nên chạy nhiều seed, ghi phiên bản thư viện/model, phần cứng, số mẫu test, và báo cáo Macro-F1 cùng Accuracy.
- LLM prompting là zero-shot, không phải fine-tuning. So sánh kết quả là so sánh chiến lược prompting với hai mô hình được huấn luyện trên tập train, không phải một so sánh đồng điều kiện hoàn toàn.
- Với PhoBERT, văn bản cần word-segment trước khi tokenization, ví dụ dùng `underthesea.word_tokenize(..., format="text")`.

## 7. Nguồn tham khảo

- UIT-VSFC dataset description: https://huggingface.co/datasets/SEACrowd/uit_vsfc
- PhoBERT model card: https://huggingface.co/vinai/phobert-base
- Hugging Face InferenceClient: https://huggingface.co/docs/huggingface_hub/package_reference/inference_client
- LLaMA 3 model card: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
