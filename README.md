# lac-hong-neural-network

Tài liệu chính của project:

- [Hướng dẫn cài đặt](SETUP.md)
- [Danh sách đề tài mini-project](LIST-DE-TAI.md)

Notebook thực hành hiện tại:

- [Buổi 1: MLP dự đoán giá nhà](notebooks/01_gia_nha_mlp.ipynb)
- [Buổi 2: Perceptron – Hamming – Hopfield](notebooks/02_perceptron_hamming_hopfield.ipynb)
- [Bài tập Buổi 2](notebooks/02_bai_tap.ipynb)
- [Buổi 3: Luật Hebb – Hopfield khử nhiễu chữ C/I/T](notebooks/03_hebb_hopfield_denoise.ipynb)
- [Bài tập Buổi 3](notebooks/03_bai_tap.ipynb)
- [Buổi 4: Perceptron từ C/I/T](notebooks/04_perceptron_cit.ipynb)
- [Bài tập Buổi 4: một node → ba đầu ra](notebooks/04_bai_tap_perceptron_cit.ipynb)
- [Bài tập ứng dụng Buổi 4: phân loại phân khúc nhà](notebooks/04_bai_tap_perceptron_phan_khuc_nha.ipynb)
- [Ngân hàng tình huống mở Buổi 4](notebooks/04_ngan_hang_tinh_huong_perceptron.ipynb)

Starter Python:

- [Buổi 4: hoàn thiện quy tắc học Perceptron](bai-tap/buoi4_perceptron_cit.py)
- [Buổi 4 mở rộng: ba Perceptron song song](bai-tap/buoi4_perceptron_cit_nhieu_dau_ra.py)
- [Buổi 4 ứng dụng: Perceptron phân loại phân khúc nhà](bai-tap/buoi4_perceptron_phan_khuc_nha.py)
- [Buổi 4: mười đề mở chỉ có comment](bai-tap/buoi4-tinh-huong/README.md)

## Luồng bài tập Buổi 4

1. Mở notebook bài tập và ghi dự đoán `net`, `ŷ` trước khi chạy.
2. Hoàn thiện 3 TODO trong starter một đầu ra, rồi chạy `run_checks()`.
3. Thử mẫu chữ C bị thiếu một nét và giải thích ảnh hưởng của feature/bias.
4. Hoàn thiện 4 TODO trong file nhiều đầu ra: ba node cùng nhận `x`, mỗi node có `W`/`b` riêng, `argmax` chọn lớp.

Chạy kiểm tra từ thư mục gốc repo:

```bash
python bai-tap/buoi4_perceptron_cit.py
python bai-tap/buoi4_perceptron_cit_nhieu_dau_ra.py
```

Hai file trên cố ý để TODO cho sinh viên; các lệnh sẽ vượt qua sau khi hoàn thiện bài.

## Case study Buổi 4 · Phân khúc nhà

Sau phần tính tay với C/I/T, sinh viên áp dụng cùng một node Perceptron vào
bài toán gần thực tế hơn: dự đoán căn nhà thuộc nhóm giá/m² cao hay phổ thông
từ vị trí, loại nhà, diện tích, số phòng, số tầng và năm xây dựng.

- Dữ liệu mô phỏng 150 căn nhà được lưu tại [`data/sample_houses.csv`](data/sample_houses.csv), chạy được offline.
- Hai cột giá chỉ dùng để tạo nhãn và bị loại khỏi feature để tránh data leakage.
- Notebook đi từ khám phá dữ liệu, train/test, huấn luyện, confusion matrix đến phân tích mẫu sai và thử căn nhà mới.
- Bài tập mở rộng so sánh các ngưỡng 80/85/90 triệu đồng/m² và ảnh hưởng của chuẩn hoá.

Chạy baseline hoàn chỉnh từ thư mục gốc repo:

```bash
python bai-tap/buoi4_perceptron_phan_khuc_nha.py
```

Xem mô tả cột và quy tắc tạo nhãn tại [`data/README.md`](data/README.md).

## Ngân hàng đề mở Buổi 4

Thư mục [`bai-tap/buoi4-tinh-huong`](bai-tap/buoi4-tinh-huong/README.md)
có mười tình huống ngắn: cảnh báo máy, tưới cây, giao hàng trễ, kiểm tra sản
phẩm, ưu tiên hỗ trợ, miễn phí vận chuyển, thông gió phòng học, lọc email rác,
nhập kho và hỗ trợ học tập. Mỗi file chỉ chứa đề bài bằng comment; sinh viên tự
chọn feature, tạo dữ liệu, viết code và đánh giá mô hình.

Notebook [`04_ngan_hang_tinh_huong_perceptron.ipynb`](notebooks/04_ngan_hang_tinh_huong_perceptron.ipynb)
tổng hợp các chủ đề và mẫu thực hành từng bước từ thu thập, xử lý dữ liệu,
huấn luyện đến triển khai thử và kiểm thử.
