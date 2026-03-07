# [HCMUT - 252] BÀI TẬP LỚN MÔN HỌC MÁY 

## 1. THÔNG TIN MÔN HỌC
- Tên môn học: Học máy
- Mã môn học:  CO3117
- Học kỳ:      HK252  
- Năm học:     2025 - 2026

## 2. GIẢNG VIÊN HƯỚNG DẪN
- Giảng viên hướng dẫn: TS.Trương Vĩnh Lân
- Thông tin liên lạc:   lantv@hcmut.edu.vn

## 3. THÔNG TIN NHÓM THỰC HIỆN
- Nhóm 10                      
- Danh sách thành viên

|     Họ và tên      |   MSSV  |            Email             | Phân công công việc|
|--------------------|---------|------------------------------|--------------------|
| Trần Gia Hân       | 2052464 | han.tran2202@hcmut.edu.vn    | 
| Lê Minh Mẫn        | 2312040 |                              
| Nguyễn Thành Trình | 2313640 |                              
| Ngô Quang Tân      | 2313052 | 
| Lương Mạnh Tiến    | 2213459 |


## 4. MỤC TIÊU BÀI TẬP LỚN
Bài tập lớn này nhằm xây dựng một pipeline học máy hoàn chỉnh cho bài toán phân loại thu nhập (dự đoán thu nhập của một cá nhân là trên hay dưới 50.000 USD/năm dựa trên các đặc trưng nhân khẩu học và nghề nghiệp), bao gồm các bước:
- Khám phá dữ liệu (EDA)
- Tiền xử lý dữ liệu
- Trích xuất / lựa chọn đặc trưng
- Huấn luyện mô hình học máy truyền thống
- Đánh giá kết quả bằng các chỉ số phù hợp
- Mở rộng với pipeline học sâu để so sánh (nếu có)

Mục tiêu cuối cùng là so sánh các cấu hình và mô hình khác nhau, phân tích ưu điểm, hạn chế và rút ra kết luận phù hợp cho bài toán.

## 5. HƯỚNG TIẾP CẬN
Nhóm triển khai dự án theo hai hướng chính:

### 5.1. Pipeline học máy truyền thống
- EDA: phân tích phân phối dữ liệu, missing value, tương quan giữa các đặc trưng
- Tiền xử lý: xử lý missing value (imputation), mã hóa biến phân loại (encoding), chuẩn 
  hóa (scaling), chia train/test
- Giảm chiều: PCA với các mức giữ lại phương sai khác nhau (90%, 95%, 99%)
- Huấn luyện và so sánh các mô hình:
  + Logistic Regression
  + Support Vector Machine (SVM)
  + Random Forest
- Đánh giá bằng các chỉ số:
  + Accuracy, Precision, Recall, F1-score
  + Confusion Matrix

### 5.2. Pipeline học sâu / trích xuất đặc trưng hiện đại


## 6. CẤU TRÚC THƯ MỤC DỰ ÁN
```text
project/
├── notebooks/
│   ├── 01_eda.ipynb                ← Phân tích khám phá dữ liệu
│   ├── 02_preprocessing.ipynb      ← Tiền xử lý dữ liệu
│   ├── 03_classical_pipeline.ipynb ← Pipeline ML truyền thống
│   ├── 04_deep_learning.ipynb      ← Pipeline học sâu
│   └── 05_main.ipynb               ← Notebook tổng hợp, chạy toàn bộ
│
├── modules/
│   ├── eda.py                      ← Hàm EDA
│   ├── preprocessing.py            ← Hàm tiền xử lý
│   ├── classical_pipeline.py       ← Hàm pipeline truyền thống
│   ├── deep_learning.py            ← Hàm pipeline học sâu
│   └── utils.py                    ← Hàm tiện ích dùng chung
│
├── reports/
│   └── report.pdf                  ← Báo cáo PDF
│
├── features/
│   └── *.npy / *.h5                ← File đặc trưng đã trích xuất
│
├── .gitignore
└── README.md

```

- Mô tả:
 + notebooks/ : chứa các notebook làm việc và notebook tổng hợp
 + modules/   : chứa các module Python hỗ trợ, được import vào notebook
 + reports/   : chứa báo cáo PDF
 + features/  : chứa các file đặc trưng đã trích xuất
 + README.md  : mô tả dự án và hướng dẫn chạy

 ## 7. HƯỚNG DẪN CHẠY

 ### 7.1. Yêu cầu môi trường

 ### 7.2.

## 8. NGUỒN DỮ LIỆU
- Tên dataset:  Adult Census Income Dataset
- Nguồn:        Kaggle / UCI Machine Learning Repository
- Link Kaggle:  https://www.kaggle.com/datasets/wenruliu/adult-income-dataset
- Link UCI:     https://archive.ics.uci.edu/dataset/2/adult
- Mô tả ngắn:   Dữ liệu điều tra dân số Mỹ (1994) gồm 48.842 mẫu, 14 đặc trưng nhân khẩu 
                học và nghề nghiệp. Bài toán phân loại nhị phân dự đoán thu nhập cá nhân trên hoặc dưới 50.000 USD/năm.

