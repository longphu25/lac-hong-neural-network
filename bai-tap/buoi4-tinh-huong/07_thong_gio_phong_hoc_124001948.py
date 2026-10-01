import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. TỰ ĐỘNG TẠO DỮ LIỆU (TARGET LIÊN TỤC 0-1)
# ==========================================
def generate_continuous_data():
    np.random.seed(42)
    n_samples = 150
    CO2 = np.random.uniform(400, 2000, n_samples)
    Temp = np.random.uniform(20, 35, n_samples)
    Humidity = np.random.uniform(40, 80, n_samples)
    People = np.random.randint(0, 50, n_samples)
    Time_Closed = np.random.uniform(0, 120, n_samples)
    
    # Tính target liên tục (Mức độ cần thông gió 0.0 -> 1.0)
    # CO2 và Temp đóng vai trò lớn nhất
    target_raw = (CO2/2000)*0.4 + (Temp/35)*0.3 + (People/50)*0.2 + (Time_Closed/120)*0.1
    # Thêm chút nhiễu (noise) thực tế và ép về khoảng [0, 1]
    target = target_raw + np.random.normal(0, 0.05, n_samples)
    target = np.clip(target, 0.0, 1.0)
    
    df = pd.DataFrame({
        'CO2': CO2, 'Temp': Temp, 'Humidity': Humidity, 
        'People': People, 'Time_Closed': Time_Closed, 'Target': target
    })
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/dataset_lien_tuc.csv', index=False)
    return df

# ==========================================
# 2. CLASS ADALINE (HỌC TARGET LIÊN TỤC)
# ==========================================
class ContinuousAdaline:
    def __init__(self, eta=0.01, n_iter=500):
        self.eta = eta
        self.n_iter = n_iter
        self.w_ = np.array([])
        self.b_ = 0.0
        self.losses_ = []

    def fit(self, X, y):
        self.w_ = np.zeros(X.shape[1])
        self.b_ = 0.0
        
        for _ in range(self.n_iter):
            net_input = np.dot(X, self.w_) + self.b_
            # Sai số liên tục e = t - net
            errors = y - net_input 
            # Cập nhật Widrow-Hoff
            self.w_ += self.eta * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * errors.mean()
            self.losses_.append((errors**2).mean()) # Ghi nhận MSE
        return self

    def predict(self, X):
        if len(self.w_) == 0:
            raise ValueError("Hãy gọi hàm fit() trước!")
        # Trả về giá trị liên tục, không dùng hàm Step >=0 nữa
        return np.dot(X, self.w_) + self.b_ 

# ==========================================
# 3. HÀM CHÍNH (MAIN THỰC THI)
# ==========================================
def main():
    print("--- 1. TẠO & ĐỌC DỮ LIỆU ---")
    df = generate_continuous_data()
    X = df[['CO2', 'Temp', 'Humidity', 'People', 'Time_Closed']].values
    y = df['Target'].values # Target từ 0.0 đến 1.0

    print("--- 2. CHIA TẬP & CHUẨN HÓA ---")
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * 0.2)
    
    X_train, X_test = X[indices[test_size:]], X[indices[:test_size]]
    y_train, y_test = y[indices[test_size:]], y[indices[:test_size]]

    mean_, std_ = np.mean(X_train, axis=0), np.std(X_train, axis=0)
    std_[std_ == 0] = 1 
    X_train_scaled = (X_train - mean_) / std_
    X_test_scaled = (X_test - mean_) / std_

    print("--- 3. HUẤN LUYỆN ADALINE LIÊN TỤC ---")
    model = ContinuousAdaline(eta=0.01, n_iter=800)
    model.fit(X_train_scaled, y_train)
    print("Huấn luyện hoàn tất! MSE cuối cùng:", model.losses_[-1])

    print("\n--- 4. PHÂN TÍCH SAI SỐ (MAX ERROR) ---")
    y_pred = model.predict(X_test_scaled)
    errors = np.abs(y_test - y_pred)
    max_err_idx = np.argsort(errors)[-2:] # Lấy 2 mẫu sai số cao nhất
    
    for i, idx in enumerate(max_err_idx):
        print(f"Mẫu lỗi lớn {i+1}: Thực tế = {y_test[idx]:.3f} | Dự đoán = {y_pred[idx]:.3f} | Lệch = {errors[idx]:.3f}")

    print("\n--- 5. KIỂM THỬ VỚI NGƯỠNG (THRESHOLD = 0.7) ---")
    def test_sensor(features, desc):
        scaled = (np.array(features) - mean_) / std_
        pred_val = model.predict(scaled)
        decision = "CẦN BẬT QUẠT 🟢" if pred_val >= 0.7 else "CHƯA CẦN 🔴"
        print(f"{desc}: Mức độ = {pred_val:.2f} -> {decision}")

    test_sensor([500, 24, 60, 5, 10], "Ca 1 (Mát mẻ)")
    test_sensor([1800, 32, 70, 40, 90], "Ca 2 (Ngột ngạt)")

if __name__ == "__main__":
    main()