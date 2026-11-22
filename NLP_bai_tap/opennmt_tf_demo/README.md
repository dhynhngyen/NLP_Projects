# OpenNMT-tf Demo: English-to-Vietnamese Machine Translation

Thực nghiệm này hướng dẫn thiết lập môi trường và sử dụng **OpenNMT-tf** để huấn luyện, đánh giá và chạy thử mô hình dịch máy **tiếng Anh → tiếng Việt** bằng kiến trúc **Transformer**.

Quy trình GPU trong tài liệu được xây dựng trên **Windows 11 + WSL2 + Ubuntu 22.04**, sử dụng GPU NVIDIA. OpenNMT-tf chạy trong Ubuntu; TensorFlow thực hiện các phép tính của mô hình và sử dụng CUDA/cuDNN để tăng tốc trên GPU.

> **Lưu ý về khả năng tái lập:** Các phiên bản phần mềm dưới đây là cấu hình của lần thực nghiệm này, không phải khẳng định rằng đây là các phiên bản mới nhất. Kết quả có thể khác nhau tùy GPU, driver, dữ liệu và cấu hình máy.

## 1. Tổng quan

### Công nghệ sử dụng

| Thành phần | Phiên bản / vai trò |
|---|---|
| Hệ điều hành chính | Windows 11 |
| Môi trường Linux | WSL2 – Ubuntu 22.04 |
| Python | 3.10 |
| TensorFlow | 2.13.1 |
| CUDA Toolkit | 11.8 |
| cuDNN | 8.6.0.163 (`nvidia-cudnn-cu11`) |
| OpenNMT-tf | 2.32.0 |
| Kiến trúc mô hình | Transformer |
| GPU thử nghiệm | NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB VRAM |

### Vai trò các thành phần

- **WSL2/Ubuntu:** cung cấp môi trường Linux để chạy các thư viện Python và TensorFlow.
- **CUDA/cuDNN:** cung cấp thư viện hỗ trợ TensorFlow thực hiện các phép tính trên GPU NVIDIA.
- **TensorFlow:** framework thực hiện tính toán mạng nơ-ron, tính loss/gradient và cập nhật trọng số.
- **OpenNMT-tf:** bộ công cụ xây dựng trên TensorFlow, hỗ trợ xử lý dữ liệu, huấn luyện, đánh giá và suy luận cho bài toán dịch máy.
- **YAML:** các file cấu hình để OpenNMT-tf biết nơi lấy dữ liệu, cách tokenization, vocabulary và thiết lập huấn luyện/đánh giá.

## 2. Cấu trúc thư mục dự án

Cấu trúc đề xuất:

```text
opennmt_tf_demo/
├── .venv/                         # Môi trường Python ảo
└── en_vi_project/
    ├── data/
    │   ├── train.en               # Câu tiếng Anh dùng để huấn luyện
    │   ├── train.vi               # Câu tiếng Việt tương ứng
    │   ├── valid.en               # Câu tiếng Anh dùng để đánh giá
    │   └── valid.vi               # Câu tiếng Việt tương ứng
    ├── tokenizer.yml              # Cấu hình tokenizer
    ├── data.yml                   # Cấu hình dữ liệu/train/eval
    ├── src-vocab.txt              # Vocabulary tiếng Anh
    ├── tgt-vocab.txt              # Vocabulary tiếng Việt
    └── run_en_vi/                 # Checkpoint, log và kết quả đánh giá
```

Các file dữ liệu song ngữ phải có cùng số dòng theo từng cặp:

- Dòng thứ *n* trong `train.en` phải là bản gốc tiếng Anh tương ứng với dòng thứ *n* trong `train.vi`.
- Dòng thứ *n* trong `valid.en` phải tương ứng với dòng thứ *n* trong `valid.vi`.

Mỗi dòng nên chứa một câu. Hãy bảo đảm dữ liệu được lưu bằng UTF-8 và đã được làm sạch theo cùng một quy tắc.

> **Dữ liệu:** README này mô tả định dạng và vị trí các file. Người chạy cần có bộ dữ liệu song ngữ Anh–Việt phù hợp và có quyền sử dụng. Nếu dữ liệu không được commit trong repository, hãy tự đặt các file vào thư mục `data/` theo đúng tên ở trên.

## 3. Chuẩn bị Windows và WSL2

### 3.1. Cài WSL2 và Ubuntu 22.04

Mở **PowerShell với quyền Administrator** trên Windows và chạy:

```powershell
wsl --install -d Ubuntu-22.04
```

Nếu WSL đã được cài đặt, có thể bỏ qua bước cài mới. Kiểm tra các bản phân phối đã cài:

```powershell
wsl -l -v
```

Đảm bảo `Ubuntu-22.04` sử dụng `VERSION 2`.

Mở Ubuntu từ Start Menu và tạo tài khoản Linux khi được yêu cầu.

### 3.2. Driver NVIDIA cho WSL2

Cài driver NVIDIA hỗ trợ WSL2 trên **Windows host** và khởi động lại máy nếu trình cài đặt yêu cầu.

Mở Ubuntu và kiểm tra GPU:

```bash
nvidia-smi
```

Kết quả cần hiển thị GPU NVIDIA của máy.

> **Quan trọng:** Với WSL2, driver NVIDIA được cài trên Windows và được WSL sử dụng. Không cài Linux display driver NVIDIA riêng bên trong Ubuntu. Khi cài CUDA Toolkit trong WSL, tránh cài các gói chung như `cuda` hoặc `cuda-drivers` nếu chúng kéo theo Linux driver; thực nghiệm này cài gói `cuda-toolkit-11-8`.

## 4. Cài CUDA Toolkit 11.8 trong Ubuntu

Các lệnh dưới đây là cách cài CUDA Toolkit 11.8 đã dùng trong thực nghiệm trên Ubuntu 22.04 WSL2.

```bash
sudo apt update
sudo apt install -y wget

wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600

wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/

sudo apt-get update
sudo apt-get install -y cuda-toolkit-11-8
```

Kiểm tra CUDA Compiler:

```bash
nvcc --version
```

Kết quả mong đợi có dòng tương tự:

```text
Cuda compilation tools, release 11.8, V11.8.89
```

`nvidia-smi` và `nvcc --version` kiểm tra hai thứ khác nhau: `nvidia-smi` kiểm tra driver/GPU nhìn thấy từ WSL; `nvcc --version` cho biết phiên bản CUDA Toolkit Compiler được cài trong Ubuntu.

## 5. Tạo môi trường Python và cài thư viện

### 5.1. Cài các gói hệ thống cần thiết

Ubuntu 22.04 thường cung cấp Python 3.10. Kiểm tra:

```bash
python3 --version
```

Cài `venv` và các công cụ cơ bản:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip build-essential
```

### 5.2. Tạo thư mục dự án và môi trường ảo

```bash
mkdir -p ~/projects/opennmt_tf_demo/en_vi_project
cd ~/projects/opennmt_tf_demo

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
```

Khi môi trường ảo được kích hoạt, dấu nhắc Terminal thường có tiền tố `(.venv)`.

### 5.3. Cài TensorFlow, cuDNN và OpenNMT-tf

Cài đúng các phiên bản đã dùng trong thực nghiệm:

```bash
python -m pip install "tensorflow==2.13.1"
python -m pip install "nvidia-cudnn-cu11==8.6.0.163"
python -m pip install "OpenNMT-tf==2.32.0"
```

Kiểm tra phiên bản:

```bash
python - <<'PY'
import tensorflow as tf
print("TensorFlow:", tf.__version__)
print("Build info:", tf.sysconfig.get_build_info())
print("GPU devices:", tf.config.list_physical_devices("GPU"))
PY

onmt-main -v
python -m pip check
```

TensorFlow 2.13.x có cấu hình build Linux tương ứng CUDA 11.8 và cuDNN 8.6. Việc cài đúng phiên bản chưa tự động đảm bảo TensorFlow nhận GPU; các thư viện động còn phải được hệ thống tìm thấy qua đường dẫn thư viện.

## 6. Thiết lập `LD_LIBRARY_PATH`

Trong lần thực nghiệm, TensorFlow đã báo `Cannot dlopen some GPU libraries` và trả về `GPU: []` trước khi thiết lập đúng đường dẫn thư viện. Sau khi bổ sung đường dẫn thư viện CUDA/cuDNN, TensorFlow nhận diện GPU.

`LD_LIBRARY_PATH` là biến môi trường cho trình nạp thư viện động của Linux biết các thư mục cần tìm file `.so`. Lệnh `export` chỉ áp dụng cho phiên shell hiện tại và các tiến trình con; do đó, nếu mở một Ubuntu Terminal mới, bạn cần thiết lập lại. Để thuận tiện, có thể khai báo một hàm trong `~/.bashrc`.

Mở file:

```bash
nano ~/.bashrc
```

Thêm đoạn sau vào cuối file (chỉnh đường dẫn nếu đặt dự án ở vị trí khác):

```bash
activate_opennmt() {
    # Kích hoạt môi trường Python của dự án
    source "$HOME/projects/opennmt_tf_demo/.venv/bin/activate" || return 1

    # Tìm các thư mục thư viện NVIDIA được cài trong môi trường Python
    local NVIDIA_LIBS
    NVIDIA_LIBS=$(python - <<'PY'
import site
import glob
import os

paths = []
for base in site.getsitepackages():
    paths.extend(glob.glob(os.path.join(base, "nvidia", "*", "lib")))

print(":".join(paths))
PY
)

    # Bổ sung đường dẫn WSL, CUDA Toolkit và các thư viện NVIDIA trong venv
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/usr/local/cuda-11.8/lib64${NVIDIA_LIBS:+:$NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    echo "OpenNMT environment activated."
    echo "LD_LIBRARY_PATH configured."
}
```

Lưu file trong nano bằng `Ctrl+O`, nhấn `Enter`, sau đó thoát bằng `Ctrl+X`.

Nạp lại cấu hình Bash và kích hoạt môi trường:

```bash
source ~/.bashrc
activate_opennmt
```

Mỗi lần mở Ubuntu Terminal mới, chỉ cần chạy:

```bash
activate_opennmt
```

Kiểm tra đường dẫn:

```bash
echo "$LD_LIBRARY_PATH"
```

> Hàm trên giả định môi trường ảo nằm tại `~/projects/opennmt_tf_demo/.venv` và CUDA Toolkit tại `/usr/local/cuda-11.8`. Nếu thay đổi vị trí hoặc phiên bản, cần cập nhật đường dẫn tương ứng. Không cần cài lại CUDA/TensorFlow mỗi lần mở Ubuntu.

## 7. Kiểm tra TensorFlow có nhận GPU không

Sau khi chạy `activate_opennmt`, thực hiện:

```bash
python - <<'PY'
import tensorflow as tf

print("TensorFlow:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPU:", gpus)

if not gpus:
    raise SystemExit("TensorFlow chưa nhận GPU. Kiểm tra driver WSL, CUDA/cuDNN và LD_LIBRARY_PATH.")

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

with tf.device("/GPU:0"):
    a = tf.random.normal([2048, 2048])
    b = tf.random.normal([2048, 2048])
    c = tf.matmul(a, b)

print("Kết quả được đặt trên:", c.device)
print("GPU test completed.")
PY
```

Nếu `GPU` có thiết bị `GPU:0` và phép nhân ma trận được đặt trên `/device:GPU:0`, TensorFlow đã nhận và thực hiện phép tính trên GPU.

Có thể mở một Ubuntu Terminal khác và theo dõi GPU trong lúc chương trình chạy:

```bash
watch -n 1 nvidia-smi
```

## 8. Chuẩn bị dữ liệu dịch Anh–Việt

Trong thư mục dự án:

```bash
cd ~/projects/opennmt_tf_demo/en_vi_project
mkdir -p data
```

Đặt bốn file dữ liệu vào `data/`:

```text
data/
├── train.en
├── train.vi
├── valid.en
└── valid.vi
```

Ý nghĩa:

| File | Nội dung |
|---|---|
| `train.en` | Câu tiếng Anh dùng để huấn luyện |
| `train.vi` | Bản dịch tiếng Việt tương ứng |
| `valid.en` | Câu tiếng Anh dùng để đánh giá |
| `valid.vi` | Bản dịch tiếng Việt chuẩn để đối chiếu |

Kiểm tra số dòng:

```bash
wc -l data/train.en data/train.vi data/valid.en data/valid.vi
```

`train.en` phải có số dòng bằng `train.vi`; `valid.en` phải có số dòng bằng `valid.vi`. Cần kiểm tra nội dung thực tế để bảo đảm các dòng cùng vị trí là cặp câu dịch tương ứng, không chỉ kiểm tra số dòng.

## 9. Cấu hình tokenizer và vocabulary

### 9.1. Tạo `tokenizer.yml`

Tạo file `tokenizer.yml` trong `en_vi_project/`:

```yaml
type: OpenNMTTokenizer
params:
  mode: aggressive
```

File này quy định OpenNMT-tf sử dụng `OpenNMTTokenizer` ở chế độ `aggressive` để tokenization văn bản. Cùng một file cấu hình được tham chiếu cho cả ngôn ngữ nguồn và ngôn ngữ đích trong thực nghiệm này.

### 9.2. Tạo vocabulary

Sau khi chuẩn bị dữ liệu và tokenizer, tạo vocabulary riêng cho tiếng Anh và tiếng Việt:

```bash
onmt-build-vocab \
  --tokenizer_config tokenizer.yml \
  --size 32000 \
  --save_vocab src-vocab.txt \
  data/train.en

onmt-build-vocab \
  --tokenizer_config tokenizer.yml \
  --size 32000 \
  --save_vocab tgt-vocab.txt \
  data/train.vi
```

Hai vocabulary phải được xây dựng với cấu hình tokenization tương ứng với cấu hình sử dụng trong huấn luyện và suy luận. Nếu repository đã cung cấp sẵn `src-vocab.txt` và `tgt-vocab.txt` được tạo từ chính dữ liệu/tokenizer của thực nghiệm, có thể bỏ qua bước tạo lại.

Có thể kiểm tra một số dòng đầu:

```bash
head -n 10 src-vocab.txt
head -n 10 tgt-vocab.txt
```

OpenNMT-tf yêu cầu vocabulary có các token đặc biệt ở đầu file; xem tài liệu Vocabulary trong phần tham khảo.

## 10. Tạo file cấu hình `data.yml`

Tạo `data.yml` trong thư mục `en_vi_project/` với nội dung:

```yaml
model_dir: run_en_vi

data:
  train_features_file: data/train.en
  train_labels_file: data/train.vi

  eval_features_file: data/valid.en
  eval_labels_file: data/valid.vi

  source_vocabulary: src-vocab.txt
  target_vocabulary: tgt-vocab.txt

  source_tokenization: tokenizer.yml
  target_tokenization: tokenizer.yml

train:
  batch_type: examples
  batch_size: 16
  effective_batch_size: 16

  max_step: 5000
  save_checkpoints_steps: 500
  keep_checkpoint_max: 3

  sample_buffer_size: 10000

eval:
  batch_size: 16
  steps: 1000
  scorers:
    - bleu
    - chrf
```

### Giải thích các thiết lập chính

| Thuộc tính | Ý nghĩa |
|---|---|
| `model_dir` | Thư mục lưu checkpoint và log của mô hình |
| `train_features_file` | Dữ liệu đầu vào huấn luyện (tiếng Anh) |
| `train_labels_file` | Dữ liệu đích huấn luyện (tiếng Việt) |
| `eval_features_file` | Dữ liệu đầu vào đánh giá (tiếng Anh) |
| `eval_labels_file` | Bản dịch chuẩn dùng để đánh giá |
| `source_vocabulary` | Vocabulary ngôn ngữ nguồn |
| `target_vocabulary` | Vocabulary ngôn ngữ đích |
| `source_tokenization` | Cấu hình tokenization cho tiếng Anh |
| `target_tokenization` | Cấu hình tokenization cho tiếng Việt |
| `batch_type: examples` | Batch được tính theo số ví dụ/cặp câu |
| `batch_size: 16` | Mỗi batch gồm 16 ví dụ |
| `effective_batch_size: 16` | Batch hiệu dụng mục tiêu là 16 |
| `max_step: 5000` | Số bước huấn luyện tối đa |
| `save_checkpoints_steps: 500` | Lưu checkpoint theo chu kỳ 500 bước |
| `keep_checkpoint_max: 3` | Giữ tối đa 3 checkpoint theo cơ chế quản lý checkpoint |
| `sample_buffer_size: 10000` | Kích thước buffer lấy mẫu/xáo trộn dữ liệu |
| `eval.steps: 1000` | Chu kỳ đánh giá theo số bước huấn luyện |
| `scorers` | Các chỉ số đánh giá: BLEU và chrF |

`--auto_config` ở lệnh chạy bên dưới yêu cầu OpenNMT-tf sử dụng các thiết lập mặc định phù hợp với loại mô hình. Có thể khai báo thêm tham số mô hình trong YAML khi cần tùy chỉnh.

## 11. Huấn luyện mô hình Transformer

Đảm bảo đã kích hoạt môi trường:

```bash
activate_opennmt
cd ~/projects/opennmt_tf_demo/en_vi_project
```

Chạy huấn luyện kết hợp đánh giá định kỳ:

```bash
onmt-main \
  --model_type Transformer \
  --config data.yml \
  --auto_config \
  train \
  --with_eval
```

Giải thích:

- `onmt-main`: chương trình dòng lệnh chính của OpenNMT-tf.
- `--model_type Transformer`: chọn kiến trúc Transformer.
- `--config data.yml`: nạp cấu hình dữ liệu và huấn luyện.
- `--auto_config`: sử dụng cấu hình tự động phù hợp với mô hình.
- `train`: thực hiện huấn luyện.
- `--with_eval`: bật đánh giá trong quá trình huấn luyện.

Checkpoint được lưu trong `run_en_vi/`. Với `save_checkpoints_steps: 500`, mô hình lưu checkpoint theo chu kỳ 500 bước.

### Theo dõi quá trình huấn luyện

Mở Terminal Ubuntu khác và chạy:

```bash
watch -n 1 nvidia-smi
```

Có thể theo dõi log bằng TensorBoard:

```bash
tensorboard --logdir=run_en_vi
```

Mở địa chỉ TensorBoard được in ra trong Terminal, thường là `http://localhost:6006`.

### Tiếp tục huấn luyện từ checkpoint

Nếu quá trình bị dừng, có thể chạy lại lệnh `train` trong cùng thư mục dự án và cùng `model_dir`. OpenNMT-tf sẽ khôi phục checkpoint gần nhất mà nó tìm thấy, nếu checkpoint tương thích với cấu hình và vocabulary đang sử dụng.

```bash
onmt-main \
  --model_type Transformer \
  --config data.yml \
  --auto_config \
  train \
  --with_eval
```

Việc khôi phục checkpoint không có nghĩa mọi bước kể từ lần lưu checkpoint gần nhất đều được giữ lại. Ví dụ, nếu checkpoint gần nhất là step 2.500 nhưng tiến trình bị dừng tại step 2.700, khi chạy lại có thể phải tiếp tục từ step 2.500.

Khi đã đạt `max_step: 5000`, muốn huấn luyện tiếp đến step 10.000 thì cần cập nhật `max_step` thành `10000`, sau đó chạy lại lệnh `train`. `max_step` là mốc step đích, không phải số step cộng thêm cho mỗi lần chạy.

## 12. Đánh giá mô hình sau huấn luyện

Chạy:

```bash
onmt-main \
  --model_type Transformer \
  --config data.yml \
  --auto_config \
  eval
```

OpenNMT-tf sử dụng dữ liệu validation đã khai báo trong `data.yml` và checkpoint trong `run_en_vi/` để đánh giá. Kết quả đánh giá và các bản dịch dự đoán được lưu trong thư mục `run_en_vi/eval/`.

### Kết quả thực nghiệm đã ghi nhận

Ở lần chạy thực nghiệm này, kết quả đánh giá tại step 5.000 là:

| Chỉ số | Kết quả |
|---|---:|
| Validation loss | 4.861225 |
| Perplexity | 129.182373 |
| BLEU | 1.175798 |
| chrF | 10.183720 |

Các điểm BLEU và chrF trên còn thấp, do đó kết quả hiện tại chỉ nên được xem là một lần chạy thử nghiệm ban đầu, chưa phải mô hình dịch máy có chất lượng sử dụng thực tế. Cần kiểm tra dữ liệu song ngữ, tokenization, vocabulary, các bản dịch dự đoán và quá trình học trước khi kết luận nguyên nhân hoặc tăng số bước huấn luyện.

## 13. Chạy thử dịch máy (Inference)

Tạo file văn bản đầu vào, ví dụ `data/test_input.en`:

```text
I love artificial intelligence.
Machine learning is an important field of computer science.
I am learning how to build neural networks.
```

Chạy lệnh dịch:

```bash
onmt-main \
  --model_type Transformer \
  --config data.yml \
  --auto_config \
  infer \
  --features_file data/test_input.en \
  --predictions_file predictions_test.vi
```

Kết quả được ghi vào:

```text
predictions_test.vi
```

Xem kết quả:

```bash
cat predictions_test.vi
```

`infer` chỉ sử dụng checkpoint để dự đoán, không cập nhật trọng số mô hình. Có thể chạy inference sau khi đã có checkpoint được lưu.

## 14. Một số lỗi thường gặp

### TensorFlow trả về `GPU: []`

Ví dụ cảnh báo:

```text
Cannot dlopen some GPU libraries.
Skipping registering GPU devices...
GPU: []
```

Kiểm tra lần lượt:

```bash
nvidia-smi
nvcc --version
echo "$LD_LIBRARY_PATH"
```

Sau đó kích hoạt lại môi trường:

```bash
activate_opennmt
```

Kiểm tra TensorFlow:

```bash
python - <<'PY'
import tensorflow as tf
print(tf.__version__)
print(tf.sysconfig.get_build_info())
print(tf.config.list_physical_devices("GPU"))
PY
```

Nếu vẫn chưa nhận GPU, cần xác định chính xác thư viện `.so` nào chưa được tìm thấy hoặc không tương thích; không nên cài lại ngẫu nhiên nhiều phiên bản CUDA/cuDNN.

### `data.yml: No such file or directory`

Nguyên nhân thường là Terminal đang ở sai thư mục. Kiểm tra:

```bash
pwd
ls
```

Sau đó di chuyển đến thư mục có `data.yml`:

```bash
cd ~/projects/opennmt_tf_demo/en_vi_project
```

### Không khôi phục được checkpoint

Kiểm tra:

```bash
cat run_en_vi/checkpoint
ls -lh run_en_vi/
```

Đảm bảo `model_dir` trong `data.yml` đúng và các file checkpoint (`.index`, `.data-*`) còn đầy đủ. Không thay đổi hoặc tạo lại vocabulary khác trong khi tiếp tục dùng checkpoint cũ nếu chưa thực hiện quy trình cập nhật vocabulary tương thích.

## 15. Khuyến nghị khi đưa dự án lên GitHub

Không nên commit môi trường ảo `.venv/` vì đây là thư mục chứa nhiều gói cài đặt theo máy. Checkpoint và dữ liệu song ngữ cũng có thể rất lớn hoặc bị giới hạn quyền chia sẻ.

Có thể dùng `.gitignore` như sau:

```gitignore
.venv/
__pycache__/
*.py[cod]
.tensorboard/
```

Nếu không muốn đưa checkpoint hoặc dữ liệu lớn lên GitHub, bổ sung:

```gitignore
run_en_vi/
data/
```

Nếu bỏ qua `data/` hoặc `run_en_vi/`, cần ghi rõ trong README rằng người dùng phải tự chuẩn bị dữ liệu và huấn luyện lại để tạo checkpoint. Không bỏ qua `src-vocab.txt`/`tgt-vocab.txt` nếu muốn người khác dùng trực tiếp đúng vocabulary đã huấn luyện; nếu không chia sẻ các file này, cần hướng dẫn tạo lại bằng đúng dữ liệu và tokenizer.

## 16. Tài liệu tham khảo

- [OpenNMT-tf 2.32 – Installation](https://opennmt.net/OpenNMT-tf/installation.html)
- [OpenNMT-tf 2.32 – Quickstart](https://opennmt.net/OpenNMT-tf/quickstart.html)
- [OpenNMT-tf – Tokenization](https://opennmt.net/OpenNMT-tf/tokenization.html)
- [OpenNMT-tf – Vocabulary](https://opennmt.net/OpenNMT-tf/vocabulary.html)
- [Microsoft – Install WSL](https://learn.microsoft.com/windows/wsl/install)
- [Ubuntu – Enable GPU acceleration on WSL](https://documentation.ubuntu.com/wsl/stable/howto/gpu-cuda/)
- [TensorFlow – Build from source / tested build configurations](https://www.tensorflow.org/install/source)

---

**Ghi chú:** Đây là hướng dẫn tái hiện cấu hình thực nghiệm OpenNMT-tf 2.32.0 + TensorFlow 2.13.1 trên Ubuntu 22.04 WSL2. Khi sử dụng GPU hoặc hệ điều hành khác, cần đối chiếu lại driver và ma trận tương thích CUDA/cuDNN chính thức.
