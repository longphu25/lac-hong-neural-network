# BUỔI 4 · TÌNH HUỐNG 10 — GỢI Ý HỖ TRỢ HỌC TẬP
#
# Bối cảnh:
# Một lớp học muốn gợi ý sinh viên nào nên được giảng viên chủ động hỗ trợ thêm.
#
# Nhãn cần dự đoán:
# 0 = theo dõi theo kế hoạch chung
# 1 = nên được mời hỗ trợ thêm
#
# Feature có thể cân nhắc:
# tỷ lệ tham dự, tỷ lệ hoàn thành bài luyện tập, số bài nộp trễ, điểm quiz gần nhất.
#
# Nhiệm vụ suy nghĩ:
# 1. Chỉ thu thập dữ liệu phục vụ học tập, tránh thông tin riêng tư hoặc nhạy cảm.
# 2. Xác định nhãn là lời mời hỗ trợ, không phải kết luận về năng lực sinh viên.
# 3. Chia train/test theo thời gian để tránh dùng dữ liệu tương lai dự đoán quá khứ.
# 4. Huấn luyện Perceptron và ưu tiên phân tích sinh viên bị bỏ sót.
# 5. Nêu vì sao kết quả mô hình phải được giảng viên xem lại trước khi hành động.
#
# Không có code mẫu hoặc lời giải trong file này.
# Sinh viên bắt đầu phần triển khai bên dưới.

