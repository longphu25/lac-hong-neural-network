# BUỔI 4 · TÌNH HUỐNG 03 — CẢNH BÁO GIAO HÀNG TRỄ
#
# Bối cảnh:
# Một cửa hàng muốn biết đơn hàng nào có nguy cơ giao trễ để chủ động thông báo.
#
# Nhãn cần dự đoán:
# 0 = có khả năng giao đúng giờ
# 1 = có nguy cơ giao trễ
#
# Feature có thể cân nhắc:
# khoảng cách, số đơn tài xế đang giữ, trời mưa, giờ cao điểm, thời gian chuẩn bị.
#
# Nhiệm vụ suy nghĩ:
# 1. Quyết định cách biến trời mưa và giờ cao điểm thành số.
# 2. Tạo dữ liệu có cả đơn dễ đoán và đơn nằm gần ranh giới quyết định.
# 3. Chia train/test trước khi huấn luyện.
# 4. Báo cáo confusion matrix thay vì chỉ báo cáo accuracy.
# 5. Chọn precision hay recall quan trọng hơn và bảo vệ lựa chọn của bạn.
#
# Không có code mẫu hoặc lời giải trong file này.
# Sinh viên bắt đầu phần triển khai bên dưới.

