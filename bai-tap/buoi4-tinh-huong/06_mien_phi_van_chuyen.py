# BUỔI 4 · TÌNH HUỐNG 06 — ĐỀ XUẤT MIỄN PHÍ VẬN CHUYỂN
#
# Bối cảnh:
# Một cửa hàng muốn tạo quy tắc đề xuất đơn hàng có được miễn phí vận chuyển.
#
# Nhãn cần dự đoán:
# 0 = khách trả phí vận chuyển
# 1 = đề xuất miễn phí vận chuyển
#
# Feature có thể cân nhắc:
# giá trị đơn hàng, khoảng cách giao, hạng thành viên, khối lượng kiện hàng.
#
# Nhiệm vụ suy nghĩ:
# 1. Xác định feature nào kéo quyết định về lớp 0 hoặc lớp 1.
# 2. Tạo các mẫu gần ranh giới, không chỉ tạo các trường hợp quá dễ.
# 3. Huấn luyện Perceptron và viết lại net của một đơn hàng bằng phép tính tay.
# 4. Thay đổi learning rate, quan sát số epoch và kết quả.
# 5. Kiểm tra quy tắc học được có tạo ra quyết định khó giải thích hay không.
#
# Không có code mẫu hoặc lời giải trong file này.
# Sinh viên bắt đầu phần triển khai bên dưới.

