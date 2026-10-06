# BUỔI 4 · TÌNH HUỐNG 02 — QUYẾT ĐỊNH TƯỚI CÂY
#
# Bối cảnh:
# Một bộ điều khiển đơn giản cần quyết định có bật máy bơm tưới cây hay không.
#
# Nhãn cần dự đoán:
# 0 = chưa cần tưới
# 1 = cần tưới
#
# Feature có thể cân nhắc:
# độ ẩm đất, nhiệt độ không khí, số giờ từ lần tưới gần nhất, khả năng có mưa.
#
# Nhiệm vụ suy nghĩ:
# 1. Xác định feature nào làm nhu cầu tưới tăng hoặc giảm.
# 2. Tự tạo tối thiểu 20 mẫu, tránh tạo tất cả mẫu theo một quy tắc quá hiển nhiên.
# 3. Dự đoán bằng tay một mẫu trước khi cho mô hình chạy.
# 4. Huấn luyện Perceptron, xem trọng số và đối chiếu với trực giác.
# 5. Đề xuất một trường hợp mà ranh giới tuyến tính có thể quyết định chưa tốt.
#
# Không có code mẫu hoặc lời giải trong file này.
# Sinh viên bắt đầu phần triển khai bên dưới.

