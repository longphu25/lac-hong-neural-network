# BUỔI 4 · TÌNH HUỐNG 08 — LỌC EMAIL RÁC ĐƠN GIẢN
#
# Bối cảnh:
# Một hộp thư nhỏ muốn đánh dấu email có khả năng là thư rác để người dùng xem lại.
#
# Nhãn cần dự đoán:
# 0 = email bình thường
# 1 = nghi ngờ là thư rác
#
# Feature có thể cân nhắc:
# số liên kết, số từ viết hoa, số dấu chấm than, độ dài tiêu đề, có từ khuyến mãi.
#
# Nhiệm vụ suy nghĩ:
# 1. Thu thập email an toàn, xoá thông tin cá nhân và ghi rõ ai là người gắn nhãn.
# 2. Chuyển nội dung email thành các feature dạng số đơn giản.
# 3. Kiểm tra hai lớp có quá mất cân bằng hay không.
# 4. Huấn luyện Perceptron và so sánh hậu quả của false positive với false negative.
# 5. Thử các email viết theo cách mới để xem mô hình có còn hoạt động hay không.
#
# Không có code mẫu hoặc lời giải trong file này.
# Sinh viên bắt đầu phần triển khai bên dưới.

