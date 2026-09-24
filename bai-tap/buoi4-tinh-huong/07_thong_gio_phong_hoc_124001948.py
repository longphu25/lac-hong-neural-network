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

import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def main():
    print("--- 1. ĐỌC VÀ TIỀN XỬ LÝ DỮ LIỆU ---")
    # Đọc file dataset từ thư mục data (theo đúng cấu trúc thư mục của anh)
    df = pd.read_csv('data/dataset.csv')
    
    # Xử lý giá trị thiếu và vượt phạm vi
    df['Humidity'] = df['Humidity'].fillna(df['Humidity'].median())
    df['CO2'] = df['CO2'].clip(lower=400, upper=5000)
    df['Temp'] = df['Temp'].clip(lower=15, upper=45)
    df['Humidity'] = df['Humidity'].clip(lower=0, upper=100)
    df['People'] = df['People'].clip(lower=0)
    df['Time_Closed'] = df['Time_Closed'].clip(lower=0)
    
    print("\n--- 2. HUẤN LUYỆN PERCEPTRON (ĐÃ CHUẨN HÓA) ---")
    X = df[['CO2', 'Temp', 'Humidity', 'People', 'Time_Closed']].values
    y = df['Label'].values
    
    # Chia tập train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Khởi tạo scaler và chuẩn hóa dữ liệu để Perceptron học tốt hơn
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Huấn luyện mô hình với số vòng lặp lớn hơn
    model = Perceptron(max_iter=2000, tol=1e-3, eta0=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"Độ chính xác trên tập test: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")

    print("--- 3. BA CA KIỂM THỬ ---")
    def predict(features):
        # Đầu vào cũng phải được chuẩn hóa qua scaler trước khi dự đoán
        features_scaled = scaler.transform([features])
        pred = model.predict(features_scaled)[0]
        return "CẦN THÔNG GIÓ (1)" if pred == 1 else "CHƯA CẦN (0)"

    print(f"Ca 1 (Bình thường - CO2: 500, Temp: 24): {predict([500, 24.0, 50.0, 10, 15])}")
    print(f"Ca 2 (Sát ngưỡng - CO2: 1150, Temp: 27): {predict([1150, 27.0, 60.0, 25, 40])}")
    # Ca 3 mô phỏng lỗi cảm biến, gán luôn bằng dữ liệu đã tiền xử lý tay cho an toàn
    print(f"Ca 3 (Lỗi cảm biến - CO2 âm): {predict([400, 45.0, 100.0, 0, 0])}")

if __name__ == "__main__":
    main()