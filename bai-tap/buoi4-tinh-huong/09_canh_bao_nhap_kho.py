# BUỔI 4 · TÌNH HUỐNG 09 — CẢNH BÁO NHẬP THÊM HÀNG
#
# Bối cảnh:
# Một cửa hàng muốn cảnh báo mặt hàng nào cần được nhập thêm trong kỳ kế tiếp.
#
# Nhãn cần dự đoán:
# 0 = chưa cần nhập thêm
# 1 = cần nhập thêm
#
# Feature có thể cân nhắc:
# số lượng tồn kho, lượng bán trung bình, số ngày chờ hàng, số đơn đang chờ.
#
# Nhiệm vụ suy nghĩ:
# 1. Chọn đơn vị quan sát: một sản phẩm trong một ngày hay trong một tuần.
# 2. Ghi lại feature tại thời điểm ra quyết định và nhãn sau một khoảng thời gian cố định.
# 3. Không dùng số lượng hết hàng trong tương lai làm feature vì sẽ gây data leakage.
# 4. Huấn luyện Perceptron và giải thích dấu của từng trọng số đã học.
# 5. Kiểm thử sản phẩm bán chậm, bán nhanh và sản phẩm mới chưa có lịch sử.
#
# Không có code mẫu hoặc lời giải trong file này.
# Sinh viên bắt đầu phần triển khai bên dưới.

