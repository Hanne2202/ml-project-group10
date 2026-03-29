# ml.project_group10
# Pipeline truyền thống
[HCMUT - 252] Machine Learning Assignment

Branch: `task/classical-pipeline`

---

## Mục tiêu branch

Triển khai và so sánh các hướng classical pipeline trên dữ liệu đã được tiền xử lý, sau đó tích hợp kết quả cuối vào `03_classical_pipeline.ipynb`.

---

## Cấu trúc thư mục

```bash
notebooks/
├── data_setup.ipynb              ← setup data dùng chung 
├── 03_classical_pipeline.ipynb   ← notebook tổng hợp cuối cùng
├── *.ipynb                    ← (tùy chọn) .
└── ...                           ← có thể thêm notebook thử nghiệm khác
````

* `data_setup.ipynb`: setup dữ liệu dùng chung cho branch này
* `03_classical_pipeline.ipynb`: notebook tổng hợp cuối cùng
* `*.ipynb`: notebook thử nghiệm riêng của từng thành viên

> Mỗi người có thể tạo notebook riêng để thử nghiệm hướng của mình. Không push thẳng kết quả chưa review vào `03_classical_pipeline.ipynb` để tránh conflict

---

## Data setup dùng chung

`data_setup.ipynb` tái tạo dữ liệu đầu vào thống nhất cho các notebook trong branch này.

Các bước chính:

* load dataset
* thay `?` thành `NaN`
* drop `education` và `fnlwgt`
* tách `X`, `y`
* train/test split với:

  * `test_size = 0.2`
  * `random_state = 42`
  * `stratify = y`
* áp dụng shared preprocessing config hiện tại: `config_2_onehot_constant_standard`

---

## Quy ước về config

Hiện tại, branch này đang dùng:

* **Shared preprocessing config hiện tại:** `config_2_onehot_constant_standard`

Lưu ý:

* Đây là **config dùng chung hiện tại**, không phải cấu hình cố định vĩnh viễn.
* Mỗi thành viên **có thể thử config khác trong notebook riêng** nếu thấy cần thiết.
* **Không tự ý sửa `data_setup.ipynb`** chỉ để phục vụ thử nghiệm cá nhân.
* Nếu nhóm **thống nhất đổi shared config**, cần cập nhật lại:

  1. `data_setup.ipynb`
  2. `README.md`

trong **cùng một commit**, đồng thời ghi rõ:

* config cũ
* config mới
* lý do thay đổi

Ví dụ commit message:

```bash
update shared preprocessing config from config_2_onehot_constant_standard to config_3_...
```

---

## Quy ước làm việc

* Dùng chung data setup để đảm bảo kết quả nhất quán
* Mỗi người có thể thử nghiệm hướng riêng trong notebook riêng
* Ghi rõ config, metric, và nhận xét trong notebook của mình
* Chỉ tích hợp vào `03_classical_pipeline.ipynb` sau khi đã review 

---

## Hướng triển khai gợi ý

Có thể triển khai theo một hoặc nhiều hướng sau:

* feature extraction / dimensionality reduction
* model training
* hyperparameter tuning
* evaluation
* summary / comparison

> Đây là hướng gợi ý chung, không bắt buộc phải đi theo một flow cứng.

---

## Quy trình làm việc trong branch này

1. Chạy hoặc copy phần setup từ `data_setup.ipynb`
2. Thử nghiệm trong notebook riêng
3. Push kết quả lên branch `task/classical-pipeline`
4. Review với các phần khác
5. Tích hợp kết quả vào `03_classical_pipeline.ipynb`

(Lưu ý không merge vào branch khác)

---

## Checklist trước khi merge vào `dev`

* [ ] Notebook thử nghiệm chạy được
* [ ] Có ghi rõ config và kết quả
* [ ] `03_classical_pipeline.ipynb` đã được cập nhật đúng
* [ ] `03_classical_pipeline.ipynb` chạy được với `Run all`
* [ ] Nếu shared config thay đổi, đã cập nhật cả `data_setup.ipynb` và `README.md`

