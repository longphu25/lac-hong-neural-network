# BUỔI 4 · TÌNH HUỐNG 07 — CẢNH BÁO THÔNG GIÓ PHÒNG HỌC
#
# Bối cảnh:
# Một lớp học muốn dùng cảm biến để cảnh báo khi cần mở cửa hoặc bật quạt thông gió.
#
# Nhãn cần dự đoán:
# 0 = chưa cần thông gió
# 1 = cần thông gió
#
# Feature có thể cân nhắc:
# nồng độ CO2, nhiệt độ, độ ẩm, số người trong phòng, thời gian đã đóng cửa.
#
# Nhiệm vụ suy nghĩ:
# 1. Xác định thời điểm và tần suất đọc cảm biến để tạo từng dòng dữ liệu.
# 2. Đề xuất cách gắn nhãn mà không dùng trực tiếp một feature làm đáp án.
# 3. Xử lý giá trị cảm biến bị thiếu, âm hoặc vượt phạm vi hợp lý.
# 4. Huấn luyện Perceptron và phân tích các mẫu nằm gần ranh giới quyết định.
# 5. Thiết kế ba ca kiểm thử: bình thường, sát ngưỡng và đầu vào không hợp lệ.
#
# Không có code mẫu hoặc lời giải trong file này.
# Sinh viên bắt đầu phần triển khai bên dưới.

