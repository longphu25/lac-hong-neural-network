# Ngân hàng tình huống Adaline · Buổi 5

Mười notebook trong thư mục này dùng lại bối cảnh của
[`bai-tap/buoi4-tinh-huong`](../../bai-tap/buoi4-tinh-huong/README.md) (Buổi 4 · Perceptron),
nhưng chuyển sang **Adaline** và quy tắc học **Widrow–Hoff (LMS)** với **target liên tục**
thay cho nhãn 0/1 cứng. Mỗi file chỉ có đề bài, khung suy nghĩ và code cell dạng comment
gợi ý; không có dữ liệu hay lời giải dựng sẵn — sinh viên tự thiết kế toàn bộ quá trình.

1. [`01_canh_bao_may_adaline.ipynb`](01_canh_bao_may_adaline.ipynb) — điểm nguy cơ máy quá tải.
2. [`02_tuoi_cay_tu_dong_adaline.ipynb`](02_tuoi_cay_tu_dong_adaline.ipynb) — mức độ / số phút cần tưới cây.
3. [`03_giao_hang_tre_adaline.ipynb`](03_giao_hang_tre_adaline.ipynb) — số phút trễ dự kiến hoặc điểm nguy cơ trễ.
4. [`04_kiem_tra_san_pham_adaline.ipynb`](04_kiem_tra_san_pham_adaline.ipynb) — điểm lỗi tổng hợp sản phẩm.
5. [`05_uu_tien_ho_tro_adaline.ipynb`](05_uu_tien_ho_tro_adaline.ipynb) — điểm mức độ ưu tiên phiếu hỗ trợ.
6. [`06_mien_phi_van_chuyen_adaline.ipynb`](06_mien_phi_van_chuyen_adaline.ipynb) — mức giảm phí vận chuyển.
7. [`07_thong_gio_phong_hoc_adaline.ipynb`](07_thong_gio_phong_hoc_adaline.ipynb) — mức độ cần thông gió.
8. [`08_loc_email_rac_adaline.ipynb`](08_loc_email_rac_adaline.ipynb) — điểm khả năng là thư rác.
9. [`09_canh_bao_nhap_kho_adaline.ipynb`](09_canh_bao_nhap_kho_adaline.ipynb) — số lượng / điểm khẩn cấp cần nhập hàng.
10. [`10_ho_tro_hoc_tap_adaline.ipynb`](10_ho_tro_hoc_tap_adaline.ipynb) — điểm mức độ cần hỗ trợ học tập.

## Khung mỗi notebook

1. Bối cảnh và feature gợi ý (giữ nguyên từ đề Buổi 4).
2. Giải thích vì sao dùng target liên tục (Adaline) thay 0/1 (Perceptron) cho tình huống đó.
3. Nhiệm vụ suy nghĩ (điều chỉnh từ đề gốc cho phù hợp Adaline).
4. Các bước triển khai còn để trống bằng comment: thu thập dữ liệu → chuẩn hoá → tách
   train/test → cài đặt & huấn luyện Adaline (Widrow–Hoff) → vẽ đường học MSE → đánh giá,
   phân tích sai số → chọn ngưỡng quyết định → kết luận.

## Yêu cầu chung gợi ý

- Chọn target là một giá trị liên tục có ý nghĩa nghiệp vụ (điểm nguy cơ, số phút, mức %...),
  không chỉ gán 0/1.
- Chuẩn hoá feature có đơn vị khác nhau trước khi huấn luyện.
- Trình bày `net = wᵀx + b`, sai số `e = t - net`, quy tắc cập nhật Widrow–Hoff và MSE qua các epoch.
- Nếu cần một quyết định rời rạc cuối cùng, chọn ngưỡng trên giá trị dự đoán liên tục và giải
  thích lựa chọn đó.
- Nêu một giới hạn của Adaline (mô hình tuyến tính) đối với tình huống đã chọn, và khi nào
  nên cân nhắc chuyển sang MLP.
