# NLP Projects

Repository này tổng hợp các bài tập thực hành, chương trình demo và các nội dung thực nghiệm liên quan đến lĩnh vực **Natural Language Processing (NLP)** trong quá trình học tập và thực hiện tiểu luận.

Các nội dung được tổ chức thành hai thư mục chính: `NLP_bai_tap` và `NLP_tieu_luan`.

## 1. Cấu trúc repository

```text
NLP_Projects/
│
├── README.md
│
├── .gitignore
├── .gitattributes
├── NLP_bai_tap/
│   ├── Demo_01/
│   ├── Demo_02/
│   ├── Demo_03/
│   └── ...
│
└── NLP_tieu_luan/
    ├── .vscode
    ├── Thuc_nghiem_Text_to_SQL
        ├── README_Text2SQL_fine_tuning
        ├── README_Text2SQL_Evaluation
        ├── Process.ipynb
        └── ...
    ├── requirements.txt
    └── ...
```

## 2. Nội dung các thư mục

### 2.1. NLP_bai_tap

Thư mục này lưu trữ các bài tập thực hành và chương trình demo được thực hiện trong quá trình tìm hiểu, học tập về xử lý ngôn ngữ tự nhiên.

Mỗi bài tập được tổ chức thành một thư mục riêng nhằm thuận tiện cho việc quản lý, chỉnh sửa và mở rộng nội dung.

Tùy theo yêu cầu của từng bài tập, thư mục có thể bao gồm:

- Mã nguồn chương trình.
- Jupyter Notebook phục vụ thực hành và thử nghiệm.
- Dữ liệu đầu vào hoặc các tệp tài nguyên cần thiết.
- File `README.md` mô tả mục tiêu, nội dung và hướng dẫn thực hiện bài tập (nếu cần).

### 2.2. NLP_tieu_luan

Thư mục này chứa các tài liệu và tệp thực nghiệm phục vụ quá trình nghiên cứu, triển khai và đánh giá nội dung tiểu luận liên quan đến lĩnh vực NLP.

Các thành phần chính có thể bao gồm:

- **Mã nguồn:** Các chương trình được xây dựng và sử dụng trong quá trình thực nghiệm.
- **Jupyter Notebook:** Các notebook phục vụ xử lý dữ liệu, thử nghiệm phương pháp và theo dõi kết quả.
- **Dữ liệu:** Các bộ dữ liệu được sử dụng trong quá trình thực nghiệm, bao gồm dữ liệu đầu vào và dữ liệu đã qua xử lý (nếu có).
- **Tệp cấu hình:** Các tệp thiết lập tham số và cấu hình phục vụ quá trình chạy chương trình.
- **Kết quả thực nghiệm:** Các kết quả, biểu đồ hoặc tài liệu cần thiết để phân tích và đánh giá mô hình.

Các thông tin chi tiết về nguồn dữ liệu, cách chuẩn bị dữ liệu, môi trường và hướng dẫn chạy thực nghiệm nên được trình bày trong file `README_Text2SQL_fine_tuning.md`  và chi tiết cách đánh giá kết quả thông qua chỉ số EX và EM được trình bày trong `README_Text2SQL_Evaluation` bên trong thư mục `Thuc_nghiem_Text_to_SQL` của thư mục `NLP_tieu_luan`.

## 3. Quản lý môi trường và dữ liệu

Các thư mục môi trường Python ảo như `venv`, `.venv`, các thư mục cache và những tệp phát sinh trong quá trình chạy chương trình không được lưu trữ trong repository.

Thay vào đó, các tệp quản lý thư viện như `requirements.txt` được sử dụng để hỗ trợ tái tạo môi trường thực nghiệm khi cần thiết.

Đối với dữ liệu có dung lượng lớn hoặc không thể phân phối trực tiếp, cần cung cấp nguồn tải và hướng dẫn chuẩn bị dữ liệu tương ứng.

## 4. Mục đích repository

Repository được xây dựng nhằm:

- Lưu trữ và quản lý tập trung các bài tập, chương trình demo và nội dung thực nghiệm NLP.
- Theo dõi quá trình học tập, nghiên cứu và phát triển các chương trình.
- Tạo điều kiện thuận lợi cho việc xem lại, chỉnh sửa và mở rộng các bài thực hành.
- Chia sẻ mã nguồn và các kết quả thực nghiệm khi cần thiết.
