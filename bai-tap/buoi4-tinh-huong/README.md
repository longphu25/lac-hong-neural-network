# Ngân hàng tình huống Perceptron · Buổi 4

Các file trong thư mục này chỉ chứa đề bài bằng comment. Không có dữ liệu mẫu,
code khởi tạo hoặc lời giải; sinh viên tự thiết kế toàn bộ quá trình thực hành.

1. [`01_canh_bao_may.py`](01_canh_bao_may.py) — cảnh báo máy quá tải.
2. [`02_tuoi_cay_tu_dong.py`](02_tuoi_cay_tu_dong.py) — quyết định tưới cây.
3. [`03_giao_hang_tre.py`](03_giao_hang_tre.py) — cảnh báo giao hàng trễ.
4. [`04_kiem_tra_san_pham.py`](04_kiem_tra_san_pham.py) — kiểm tra sản phẩm.
5. [`05_uu_tien_ho_tro.py`](05_uu_tien_ho_tro.py) — ưu tiên phiếu hỗ trợ.
6. [`06_mien_phi_van_chuyen.py`](06_mien_phi_van_chuyen.py) — miễn phí vận chuyển.
7. [`07_thong_gio_phong_hoc.py`](07_thong_gio_phong_hoc.py) — cảnh báo thông gió phòng học.
8. [`08_loc_email_rac.py`](08_loc_email_rac.py) — lọc email rác đơn giản.
9. [`09_canh_bao_nhap_kho.py`](09_canh_bao_nhap_kho.py) — cảnh báo nhập thêm hàng.
10. [`10_ho_tro_hoc_tap.py`](10_ho_tro_hoc_tap.py) — gợi ý hỗ trợ học tập.

Notebook tổng hợp: [`../../notebooks/04_ngan_hang_tinh_huong_perceptron.ipynb`](../../notebooks/04_ngan_hang_tinh_huong_perceptron.ipynb).

## Yêu cầu chung gợi ý

- Mỗi nhóm chọn một tình huống và tự xác định câu hỏi phân loại nhị phân.
- Tự tạo hoặc thu thập dữ liệu nhỏ, ghi rõ ý nghĩa và đơn vị từng feature.
- Tách train/test; không đưa thông tin chứa trực tiếp đáp án vào feature.
- Trình bày `net = wᵀx + b`, quy tắc ngưỡng và ít nhất một lượt cập nhật.
- Báo cáo confusion matrix và phân tích tối thiểu hai mẫu dự đoán sai.
- Nêu một giới hạn của Perceptron đối với tình huống đã chọn.

Notebook tổng hợp cung cấp mẫu thực hành từng bước từ kế hoạch thu thập dữ liệu,
xử lý dữ liệu, huấn luyện đến triển khai thử và kiểm thử. Các code cell vẫn chỉ
chứa yêu cầu dạng comment để sinh viên tự viết phần triển khai.
