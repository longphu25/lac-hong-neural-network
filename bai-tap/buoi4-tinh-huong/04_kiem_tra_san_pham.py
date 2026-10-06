# BUỔI 4 · TÌNH HUỐNG 04 — KIỂM TRA SẢN PHẨM CUỐI DÂY CHUYỀN
#
# Bối cảnh:
# Một dây chuyền cần phân loại nhanh sản phẩm đạt hay cần chuyển sang kiểm tra lại.
#
# Nhãn cần dự đoán:
# 0 = đạt yêu cầu
# 1 = cần kiểm tra lại
#
# Feature có thể cân nhắc:
# độ lệch khối lượng, độ lệch kích thước, số vết xước, điểm màu sắc.
#
# Nhiệm vụ suy nghĩ:
# 1. Dùng độ lệch tuyệt đối thay vì kích thước thô nếu cả quá lớn và quá nhỏ đều xấu.
# 2. Tự tạo dữ liệu nhỏ và mô tả rõ đơn vị của từng cột.
# 3. Huấn luyện Perceptron và tìm các sản phẩm bị phân loại sai.
# 4. Đọc dấu của trọng số; kiểm tra nó có hợp lý với nghiệp vụ không.
# 5. Thử thay đổi ngưỡng gán nhãn và ghi lại ảnh hưởng lên hai lớp.
#
# Không có code mẫu hoặc lời giải trong file này.
# Sinh viên bắt đầu phần triển khai bên dưới.

