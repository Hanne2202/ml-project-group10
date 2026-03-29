# ml.project_group10
[HCMUT - 252] Machine Learning Assignment

Branch: `dev`

---

## Giới thiệu

Đây là branch tích hợp chính của nhóm trong quá trình thực hiện bài tập lớn môn Machine Learning.

Hiện tại, nhóm đã hoàn thành:
- **EDA (Exploratory Data Analysis)**
- **Data Preprocessing**

Các phần tiếp theo sẽ được phát triển và tích hợp dần trên branch này, bao gồm:
- **Classical / Traditional Pipeline**
- **Deep Learning Pipeline**
- **Final comparison & summary**

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

### Đang triển khai
- Classical / Traditional Pipeline
- Deep Learning Pipeline

### Sẽ thực hiện tiếp
- So sánh kết quả giữa các hướng tiếp cận
- Tổng hợp kết quả cuối cùng
- Hoàn thiện notebook / báo cáo cuối kỳ

---

## Cấu trúc thư mục

```bash
notebooks/
├── 01_eda.ipynb
├── 02_preprocessing.ipynb
├── 03_classical_pipeline.ipynb
├── 04_deep_learning.ipynb
└── 05_main.ipynb
````

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

* Các notebook có thể tiếp tục được cập nhật trong quá trình nhóm hoàn thiện project
* README này sẽ được điều chỉnh thêm khi branch `dev` tích hợp xong phần classical pipeline và deep learning pipeline

