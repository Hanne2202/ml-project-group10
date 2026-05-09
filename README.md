# ml-project-group10

[HCMUT - 252] Machine Learning Assignment

Branch tích hợp chính: `dev`

---

## Giới thiệu

Đây là repository tích hợp chính của nhóm trong quá trình thực hiện bài tập lớn môn Machine Learning.

Đã có đầy đủ các phần: **EDA**, **tiền xử lý**, **pipeline classical**, **pipeline deep learning (MLP / TabNet)**, và **notebook tổng hợp** trong thư mục `notebooks/`. Chi tiết từng notebook xem phần [Cấu trúc thư mục](#cấu-trúc-thư-mục).

---

## Bài toán

Nhóm thực hiện bài toán **phân loại thu nhập** trên bộ dữ liệu **Adult Income Dataset**.

- **Input:** các đặc trưng nhân khẩu học và nghề nghiệp
- **Output:** dự đoán mức thu nhập
  - `<=50K`
  - `>50K`

---

## Tiến độ hiện tại

### Đã hoàn thành
- Khảo sát dữ liệu ban đầu
- Kiểm tra missing value và giá trị bất thường
- Phân tích phân phối target
- Phân tích một số đặc trưng số và đặc trưng phân loại quan trọng
- Chuẩn hóa missing value
- Loại bỏ một số cột không cần thiết / trùng lặp
- Xây dựng và so sánh nhiều cấu hình preprocessing
- Chọn shared preprocessing config hiện tại để dùng cho các bước tiếp theo

### Đã có trong notebook
- Classical / Traditional Pipeline (`03_classical_pipeline.ipynb`)
- Deep Learning Pipeline — MLP và TabNet (`04_deep_learning.ipynb`)
- Notebook tổng hợp (`05_main.ipynb`)

### Sẽ chỉnh sửa / bổ sung khi cần
- So sánh và trình bày kết quả trong báo cáo cuối kỳ
- Tinh chỉnh thử nghiệm (ví dụ bật lại các cell Optuna tốn thời gian)

---

## Yêu cầu

- **Python:** khuyến nghị 3.10, 3.11 hoặc 3.12 (tương thích PyTorch và scikit-learn).
- **RAM / thời gian chạy:** notebook deep learning và TabNet tốn tài nguyên hơn classical ML; có thể chạy trên CPU (chậm hơn GPU).
- **Mạng:** notebook EDA và các notebook sau tải dữ liệu từ URL công khai (GitHub); cần kết nối Internet khi chạy lần đầu.

---

## Cài đặt và chạy

### 1. Clone repository

```bash
git clone https://github.com/Hanne2202/ml-project-group10.git
cd ml-project-group10
```

### 2. Tạo môi trường ảo (khuyến nghị)

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Cài các thư viện

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**PyTorch với GPU (CUDA):** `pip install -r requirements.txt` thường cài bản PyTorch hỗ trợ CPU. Nếu cần GPU, cài trước PyTorch (build CUDA) theo lệnh tại [pytorch.org](https://pytorch.org/get-started/locally/), sau đó cài phần còn lại; nếu `pip` cố ghi đè PyTorch, tạm thời bỏ hoặc comment dòng `torch` trong `requirements.txt` rồi chạy lại `pip install -r requirements.txt`.

### 4. Khởi chạy Jupyter

Từ thư mục gốc của project:

```bash
jupyter notebook
```

hoặc dùng JupyterLab / VS Code mở từng file `.ipynb` trong `notebooks/`.

### 5. Thứ tự và cách chạy notebook

- Các notebook gắn thư mục `modules/` bằng `sys.path.insert(0, os.path.abspath('..'))`, nghĩa là **`modules` là gói Python ở cấp cha của `notebooks/`**. Hãy mở hoặc chạy notebook khi working directory là `notebooks/` (cách Jupyter mặc định khi mở file trong thư mục đó) để đường dẫn tương đối đúng.
- **Thứ tự khuyến nghị:**
  1. `01_eda.ipynb` — khám phá dữ liệu  
  2. `02_preprocessing.ipynb` — tiền xử lý, chọn shared config  
  3. `03_classical_pipeline.ipynb` — mô hình truyền thống  
  4. `04_deep_learning.ipynb` — MLP, TabNet, Optuna (một số cell Optuna được mock / comment để chạy nhanh; có thể bật lại để chạy đầy đủ, thời gian lâu)  
  5. `05_main.ipynb` — tổng hợp pipeline (tuỳ mục đích báo cáo)

Notebook `05_main.ipynb` có thể chạy độc lập nếu đã có đầy đủ thư viện và dữ liệu tải được; các notebook khác nên đọc theo thứ tự để hiểu luồng xử lý.

### 6. Khởi chạy ứng dụng Web (Streamlit)

Nhóm đã phát triển giao diện người dùng dựa trên Streamlit để dự đoán trực tiếp từ trình duyệt. Tại thư mục gốc của dự án, hãy chạy:

```bash
streamlit run app.py
```
Sau đó truy cập đường dẫn `http://localhost:8501` hiển thị trên terminal để sử dụng ứng dụng.

---

## Cấu trúc thư mục

```text
ml-project-group10/
├── modules/                    # Logic Python dùng chung cho notebook
│   ├── eda.py
│   ├── preprocessing.py
│   ├── classical_learning.py
│   ├── deep_learning.py
│   ├── tuning.py
│   └── mappings.py             
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_classical_pipeline.ipynb
│   ├── 04_deep_learning.ipynb
│   └── 05_main.ipynb
├── app.py                      
├── requirements.txt
└── README.md
```

### Mô tả các notebook

* `01_eda.ipynb`: phân tích khám phá dữ liệu
* `02_preprocessing.ipynb`: tiền xử lý dữ liệu và so sánh các cấu hình preprocessing
* `03_classical_pipeline.ipynb`: triển khai pipeline truyền thống
* `04_deep_learning.ipynb`: triển khai pipeline học sâu
* `05_main.ipynb`: notebook tổng hợp / kết quả cuối cùng

> Lưu ý: một số notebook có thể đang được hoàn thiện dần và sẽ tiếp tục được cập nhật trên branch `dev`.

---

## EDA

Notebook `01_eda.ipynb` tập trung vào:

* tổng quan dữ liệu
* kiểu dữ liệu và thống kê mô tả
* kiểm tra missing value
* phân phối biến mục tiêu
* phân tích các đặc trưng quan trọng theo target
* kiểm tra một số biến có outlier hoặc phân phối lệch

EDA được dùng để hỗ trợ các quyết định ở bước preprocessing và định hướng cho phần modeling.

---

## Preprocessing

Notebook `02_preprocessing.ipynb` hiện đang là cơ sở cho phần modeling tiếp theo.

Các bước chính:

* thay thế `?` bằng `NaN`
* loại bỏ:

  * `education`
  * `fnlwgt`
* tách `X` và `y`
* chia train/test
* xác định numerical features và categorical features
* xây dựng nhiều cấu hình preprocessing
* so sánh các cấu hình để chọn shared config hiện tại

### Shared preprocessing config hiện tại

`config_2_onehot_constant_standard`

Cấu hình này hiện được dùng làm mốc chung cho các bước triển khai tiếp theo. Nếu sau này nhóm thống nhất đổi config, cần cập nhật lại notebook liên quan và ghi rõ trong commit.

---

## Hướng phát triển tiếp theo

### 1. Classical / Traditional Pipeline

* huấn luyện và đánh giá các mô hình học máy truyền thống
* tuning và so sánh kết quả
* phân tích hiệu năng giữa các cấu hình / mô hình

### 2. Deep Learning Pipeline

* xây dựng mô hình học sâu phù hợp với dữ liệu tabular
* huấn luyện, theo dõi loss / metric
* đánh giá và so sánh với pipeline truyền thống

### 3. Final Summary

* tổng hợp kết quả
* so sánh hai hướng tiếp cận
* rút ra kết luận cuối cùng

---

## Quy ước làm việc trên branch `dev`

* `dev` là branch tích hợp chung của nhóm
* các thay đổi lớn nên được thực hiện trước trên branch nhiệm vụ riêng, sau đó mới merge vào `dev`
* không chỉnh sửa trực tiếp các file tổng hợp khi chưa thống nhất
* khi update notebook, nên đảm bảo notebook có thể chạy lại hợp lý
* khi shared config thay đổi, cần ghi rõ trong commit / pull request

---

## Ghi chú

- Cập nhật shared preprocessing (`config_*` trong `02_preprocessing.ipynb`) thì nhớ đồng bộ các notebook phía sau và ghi rõ trong commit.
- Cell chạy Optuna / huấn luyện dài trong `04_deep_learning.ipynb` và `05_main.ipynb` có thể được giữ mock hoặc comment để chạy nhanh; khi báo cáo cần số đo mới, bật chạy lại và chờ đủ thời gian.

