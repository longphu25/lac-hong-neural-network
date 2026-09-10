# BUỔI 4 · TÌNH HUỐNG 05 — ƯU TIÊN PHIẾU HỖ TRỢ KHÁCH HÀNG
#
# Bối cảnh:
# Một nhóm hỗ trợ muốn đưa các phiếu khẩn cấp lên đầu hàng đợi.
#
# Nhãn cần dự đoán:
# 0 = xử lý theo hàng đợi thông thường
# 1 = cần ưu tiên
#
# Feature có thể cân nhắc:
# số từ khẩn cấp, dịch vụ có bị gián đoạn hay không, thời gian chờ, mức khách hàng.
#
# Nhiệm vụ suy nghĩ:
# 1. Chỉ dùng feature đo được tại thời điểm phiếu vừa được gửi.
# 2. Giải thích feature nào có nguy cơ gây thiên lệch hoặc không công bằng.
# 3. Tạo dữ liệu nhỏ, mã hoá feature phân loại và chuẩn hoá khi cần.
# 4. So sánh lỗi bỏ sót phiếu khẩn cấp với lỗi ưu tiên nhầm.
# 5. Nêu một quyết định vẫn cần con người xem lại thay vì giao hoàn toàn cho mô hình.
#
# Không có code mẫu hoặc lời giải trong file này.
# Sinh viên bắt đầu phần triển khai bên dưới.

