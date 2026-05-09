BÀI TẬP LỚN MÔN HỌC MÁY - HK 252
> # ml-project-group10
> **Môn học:** Học Máy — **Mã môn:** CO3117 — **Học kỳ:** 252 — **Năm học:** 2025–2026

---

## Thông tin nhóm

| Họ và tên | MSSV | Email | Ghi chú |
|---|---|---|---|
| Trần Gia Hân | 2052464 | han.tran2202@hcmut.edu.vn | Nhóm trưởng |
| Lê Minh Mẫn | 2312040 | man.le9905@hcmut.edu.vn | |
| Nguyễn Thành Trình | 2313640 | trinh.nguyen020705@hcmut.edu.vn | |
| Ngô Quang Tân | 2313052 | tan.ngo196@hcmut.edu.vn | |
| Lương Mạnh Tiến | 2213459 | | |

**Giảng viên hướng dẫn (GVHD):** Trường Vĩnh Lân  
**Lớp:** L01 — **Nhóm:** 10

---

## Tài nguyên

| Tài nguyên | Đường dẫn |
|---|---|
| 📄 Báo cáo PDF | [final\_report.pdf](https://drive.google.com/file/d/1VwfrjaARU0-Mgm6kccpsh-jXPsgWhNKY/view?usp=sharing) |
| 🔗 Google Colab | [BTL\_Machine\_Learning.ipynb](https://colab.research.google.com/drive/1LskLJh_-S8esA2RgfWagCwlj2WNw-m_0?usp=sharing) |

---

## Bài toán

Nhóm thực hiện bài toán **phân loại thu nhập** (binary classification) trên bộ dữ liệu **Adult Census Income**.

- **Input:** các đặc trưng nhân khẩu học và nghề nghiệp (tuổi, học vấn, tình trạng hôn nhân, nghề nghiệp, số giờ làm việc, ...)
- **Output:** dự đoán mức thu nhập cá nhân mỗi năm
  - `<=50K`
  - `>50K`

---

## Mục tiêu bài tập lớn

1. **Xây dựng pipeline học máy hoàn chỉnh** — từ khám phá dữ liệu (EDA), tiền xử lý, trích xuất đặc trưng, huấn luyện, tinh chỉnh siêu tham số đến đánh giá và triển khai.
2. **So sánh nhiều mô hình trên cùng bộ dữ liệu** — Logistic Regression, Linear SVM, Random Forest, Gaussian Naive Bayes (pipeline truyền thống) và MLP, TabNet (pipeline học sâu).
3. **Phân tích ảnh hưởng của tiền xử lý** — so sánh các cấu hình preprocessing khác nhau, lựa chọn cấu hình tốt nhất dựa trên thực nghiệm.
4. **Đánh giá tính phù hợp của học sâu trên dữ liệu bảng** — kiểm chứng xem MLP hay TabNet có thực sự vượt trội hơn các mô hình truyền thống không.
5. **Tăng khả năng diễn giải và tái sử dụng** — phân tích feature importance, tổ chức code dạng module, triển khai ứng dụng web bằng Streamlit.

---

## Kết quả tóm tắt

| Mô hình | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression (baseline) | 0.8074 | 0.6760 | 0.9051 |
| **Random Forest (tuned)** ✅ | **0.8341** | **0.7050** | **0.9175** |
| MLP (Optuna) | 0.8101 | 0.6919 | 0.9102 |
| TabNet (Optuna) | 0.8136 | 0.6835 | 0.9029 |

**Mô hình tốt nhất:** Random Forest (tuned) — pipeline học máy truyền thống với F1-Score = 0.7050, ROC-AUC = 0.9175.

---

## Yêu cầu hệ thống

- **Python:** 3.10, 3.11 hoặc 3.12 (khuyến nghị)
- **RAM:** tối thiểu 4 GB; notebook deep learning và TabNet tốn nhiều tài nguyên hơn
- **GPU (tuỳ chọn):** hỗ trợ CUDA để tăng tốc huấn luyện MLP/TabNet
- **Kết nối Internet:** cần thiết khi tải dữ liệu từ URL công khai lần đầu

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

### 3. Cài đặt thư viện

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **PyTorch với GPU (CUDA):** `pip install -r requirements.txt` mặc định cài bản CPU.  
> Nếu cần GPU, cài PyTorch (CUDA build) theo hướng dẫn tại [pytorch.org](https://pytorch.org/get-started/locally/) trước, sau đó comment dòng `torch` trong `requirements.txt` rồi chạy lại lệnh cài.

### 4. Khởi chạy Jupyter

```bash
jupyter notebook
```

Hoặc mở từng file `.ipynb` trong `notebooks/` bằng JupyterLab / VS Code.

### 5. Thứ tự chạy notebook

> Các notebook sử dụng `sys.path.insert(0, os.path.abspath('..'))` để import từ `modules/`. Hãy đảm bảo working directory là `notebooks/` khi chạy (mặc định của Jupyter khi mở file trong thư mục đó).

| Thứ tự | Notebook | Nội dung |
|---|---|---|
| 1 | `01_eda.ipynb` | Khám phá và phân tích dữ liệu |
| 2 | `02_preprocessing.ipynb` | Tiền xử lý và so sánh cấu hình preprocessing |
| 3 | `03_classical_pipeline.ipynb` | Pipeline học máy truyền thống |
| 4 | `04_deep_learning.ipynb` | Pipeline học sâu (MLP, TabNet, Optuna) |
| 5 | `05_main.ipynb` | Notebook tổng hợp — kết quả cuối cùng |

> **Lưu ý:** Notebook `05_main.ipynb` có thể chạy độc lập nếu đã có đủ thư viện và dữ liệu. Một số cell Optuna/TabNet trong `04_deep_learning.ipynb` được mock/comment để chạy nhanh — bật lại khi cần số liệu chính xác.

### 6. Chạy ứng dụng Web (Streamlit)

```bash
streamlit run app.py
```

Sau đó truy cập `http://localhost:8501` để sử dụng giao diện dự đoán thu nhập trực tiếp.

> **Lưu ý:** Ứng dụng yêu cầu các file model đã được lưu trong `models/`. Nếu thiếu, hãy chạy `04_deep_learning.ipynb` trước để sinh các file cần thiết.

---

## Cấu trúc thư mục

```text
ml-project-group10/
├── data/
│   └── adult.csv                   # Bộ dữ liệu Adult Census Income
├── features/                       # Đặc trưng trung gian được lưu lại
│   └── .gitkeep
├── models/                         # Model và các bộ tiền xử lý đã lưu
│   ├── best_dl_model.pth           # Model học sâu tốt nhất (PyTorch)
│   ├── best_f1_score.txt           # F1-score của model tốt nhất
│   ├── best_model_type.txt         # Loại model tốt nhất
│   ├── ohe_encoder.joblib          # Bộ mã hóa One-Hot Encoder
│   └── scaler.joblib               # Bộ chuẩn hóa StandardScaler
├── modules/                        # Logic Python dùng chung cho notebook
│   ├── eda.py                      # Hàm phân tích khám phá dữ liệu
│   ├── preprocessing.py            # Pipeline tiền xử lý dữ liệu
│   ├── classical_learning.py       # Huấn luyện và đánh giá mô hình truyền thống
│   ├── deep_learning.py            # Mô hình MLP và TabNet
│   ├── tuning.py                   # Tối ưu siêu tham số bằng Optuna
│   └── mappings.py                 # Ánh xạ nhãn và cấu hình dùng chung
├── notebooks/
│   ├── 01_eda.ipynb                # Phân tích khám phá dữ liệu
│   ├── 02_preprocessing.ipynb      # Tiền xử lý và so sánh cấu hình
│   ├── 03_classical_pipeline.ipynb # Pipeline học máy truyền thống
│   ├── 04_deep_learning.ipynb      # Pipeline học sâu (MLP, TabNet, Optuna)
│   └── 05_main.ipynb               # Notebook tổng hợp / kết quả cuối cùng
├── reports/
│   ├── final/                      # Báo cáo cuối kỳ
│   └── progress/                   # Báo cáo tiến độ
├── app.py                          # Ứng dụng web Streamlit
├── requirements.txt                # Danh sách thư viện cần cài
└── README.md
```

---

## Mô tả chi tiết các module

### `modules/eda.py`
Các hàm hỗ trợ phân tích khám phá dữ liệu (EDA): thống kê mô tả, trực quan hóa phân phối biến mục tiêu, phân tích giá trị thiếu, phân tích biến số và biến phân loại.

### `modules/preprocessing.py`
Xây dựng pipeline tiền xử lý bằng `Pipeline` và `ColumnTransformer` của scikit-learn. Hỗ trợ nhiều cấu hình (chuẩn hóa, mã hóa one-hot, xử lý missing value), chia tập train/test có stratify và xuất shared preprocessing config.

### `modules/classical_learning.py`
Huấn luyện và đánh giá các mô hình học máy truyền thống: Logistic Regression, Linear SVM, Random Forest, Gaussian Naive Bayes. Bao gồm baseline training, GridSearchCV, RandomizedSearchCV và phân tích feature importance.

### `modules/deep_learning.py`
Định nghĩa kiến trúc MLP (PyTorch) và tích hợp TabNet (pytorch-tabnet). Bao gồm vòng huấn luyện, early stopping, learning rate scheduling và lưu model.

### `modules/tuning.py`
Tối ưu siêu tham số bằng Optuna (TPESampler + MedianPruner) cho cả MLP và TabNet. Hỗ trợ pretraining tự giám sát cho TabNet.

### `modules/mappings.py`
Ánh xạ nhãn và các hằng số cấu hình dùng chung giữa notebook và ứng dụng Streamlit.

---

## EDA — Những phát hiện chính

- **Mất cân bằng lớp:** ~76% mẫu thuộc nhóm `<=50K`, chỉ ~24% thuộc nhóm `>50K` → ưu tiên F1-score thay vì Accuracy.
- **Giá trị thiếu:** ký hiệu `?` xuất hiện ở `workclass` (2,799 mẫu), `occupation` (2,809 mẫu), `native-country` (857 mẫu).
- **Đặc trưng dư thừa:** `education` và `educational-num` phản ánh cùng thông tin; `fnlwgt` là trọng số điều tra không liên quan trực tiếp đến bài toán → loại bỏ cả hai.
- **Biến số lệch phân phối:** `capital-gain` và `capital-loss` có phân phối lệch phải mạnh (phần lớn = 0).

---

## Tiền xử lý — Cấu hình được chọn

Nhóm so sánh 4 cấu hình preprocessing bằng Logistic Regression, lấy F1-score làm tiêu chí chọn:

| Cấu hình | Missing categorical | Biến số | F1-score | Số đặc trưng |
|---|---|---|---|---|
| **C2** ✅ | Điền hằng `Missing` | StandardScaler | **0.6649** | 91 |
| C1 | Điền most frequent | StandardScaler | 0.6578 | 88 |
| C3 | Điền most frequent | MinMaxScaler | 0.6553 | 88 |
| C4 | Điền most frequent | Không chuẩn hóa | 0.6583 | 88 |

**Cấu hình được chọn:** `config_2_onehot_constant_standard` (91 đặc trưng sau One-Hot Encoding).

---

## Quy ước làm việc trên branch `dev`

- `dev` là branch tích hợp chung của nhóm.
- Thay đổi lớn nên thực hiện trên branch nhiệm vụ riêng, sau đó mới merge vào `dev`.
- Không sửa trực tiếp các file tổng hợp khi chưa thống nhất trong nhóm.
- Khi cập nhật notebook, đảm bảo notebook có thể chạy lại hợp lý từ đầu.
- Khi shared config thay đổi, ghi rõ trong commit message và đồng bộ các notebook liên quan.


