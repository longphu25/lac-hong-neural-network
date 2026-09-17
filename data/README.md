# Dữ liệu thực hành

## `sample_houses.csv`

Bộ dữ liệu mô phỏng gồm 150 căn nhà tại Hà Nội và TP. Hồ Chí Minh, được lưu
ngay trong repo để sinh viên có thể huấn luyện và chạy thử hoàn toàn offline.
Dữ liệu chỉ phục vụ giảng dạy, không dùng để tư vấn hoặc định giá bất động sản.

| Cột | Ý nghĩa | Đơn vị / kiểu |
| --- | --- | --- |
| `id` | Mã mẫu | số nguyên |
| `city` | Thành phố | phân loại |
| `district` | Quận / khu vực | phân loại |
| `area_m2` | Diện tích | m² |
| `bedrooms` | Số phòng ngủ | số nguyên |
| `bathrooms` | Số phòng tắm | số nguyên |
| `floors` | Số tầng | số nguyên |
| `building_type` | Loại hình nhà | `apartment`, `house`, `townhouse` |
| `year_built` | Năm xây dựng | năm |
| `price_million_vnd` | Tổng giá | triệu đồng |
| `price_per_m2_million_vnd` | Giá trên mỗi m² | triệu đồng/m² |

## Cách dùng trong Buổi 4

Bài tập tạo nhãn nhị phân như sau:

```text
phan_khuc_cao = 1 nếu price_per_m2_million_vnd >= 85
phan_khuc_cao = 0 nếu price_per_m2_million_vnd < 85
```

Mô hình chỉ nhận vị trí, loại nhà, diện tích, số phòng, số tầng và năm xây dựng.
Không đưa `price_million_vnd` hoặc `price_per_m2_million_vnd` vào feature: hai
cột này chứa trực tiếp thông tin dùng để tạo nhãn, nên sử dụng chúng sẽ gây
**rò rỉ dữ liệu (data leakage)** và làm kết quả đánh giá tốt một cách giả tạo.

File thực hành liên quan:

- [`../bai-tap/buoi4_perceptron_phan_khuc_nha.py`](../bai-tap/buoi4_perceptron_phan_khuc_nha.py)
- [`../notebooks/04_bai_tap_perceptron_phan_khuc_nha.ipynb`](../notebooks/04_bai_tap_perceptron_phan_khuc_nha.ipynb)
