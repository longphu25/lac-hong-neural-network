import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

# ==========================================
# 1. HÀM TỰ ĐỘNG DÒ TÌM ĐƯỜNG DẪN DỮ LIỆU
# ==========================================
def find_data_file(filename):
    possible_paths = [
        os.path.join(os.getcwd(), 'data', filename),
        os.path.join(os.getcwd(), '..', 'data', filename),
        os.path.join(os.getcwd(), 'lac-hong-neural-network-buoi-4', 'data', filename)
    ]
    
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths.append(os.path.join(current_dir, '..', 'data', filename))
    except NameError:
        pass
        
    for path in possible_paths:
        path = os.path.normpath(path)
        if os.path.exists(path):
            return path
            
    return None

# ==========================================
# 2. XÂY DỰNG MÔ HÌNH PERCEPTRON
# ==========================================
class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self._step_function(linear_output)
                
                update = self.lr * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self._step_function(linear_output)

    def _step_function(self, x):
        return np.where(x >= 0, 1, 0)

# ==========================================
# 3. HÀM THỰC THI CHÍNH
# ==========================================
def main():
    filename = 'industrial_fault_detection_data_1000.csv'
    
    data_path = find_data_file(filename)
    
    if data_path is None:
        print(f"\n[!] LỖI: Không tìm thấy file '{filename}'.")
        sys.exit(1)

    print(f"\n[*] Đã tìm thấy dữ liệu tại:\n    -> {data_path}")
    
    # Đọc dữ liệu
    df = pd.read_csv(data_path)
    
    # Chỉ giữ lại các cột số để tránh lỗi chuỗi/thời gian
    df_numeric = df.select_dtypes(include=[np.number])
    
    # Chọn Features và Label
    X = df_numeric.iloc[:, [0, 1]].values
    y = df_numeric.iloc[:, -1].values
    
    # BƯỚC SỬA LỖI QUAN TRỌNG: ÉP VỀ PHÂN LOẠI NHỊ PHÂN
    # Nếu y = 0 -> Bình thường (0)
    # Nếu y > 0 (1, 2, 3...) -> Lỗi (1)
    y = np.where(y > 0, 1, 0)
    
    print(f"[*] Dữ liệu đã được đưa về dạng nhị phân (0: Bình thường, 1: Có lỗi).")
    
    # Tiền xử lý
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Huấn luyện mô hình
    print("[*] Đang huấn luyện Perceptron...")
    model = Perceptron(learning_rate=0.01, n_iters=1000)
    model.fit(X_train_scaled, y_train)
    
    # Đánh giá
    predictions = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    
    print(f"\n====================================")
    print(f"[*] Độ chính xác (Accuracy): {acc * 100:.2f}%")
    print(f"[*] Ma trận nhầm lẫn (Confusion Matrix):\n{cm}")
    
    # Phân tích bài toán
    print("\n--- PHÂN TÍCH TÌNH HUỐNG THỰC TẾ ---")
    if cm.shape == (2, 2):
        fn = cm[1, 0] # Dòng 1 (Thực tế là 1), Cột 0 (Đoán là 0)
        print(f"Số lượng False Negative (Máy QUÁ TẢI nhưng mô hình đoán BÌNH THƯỜNG): {fn}")
        print("=> Đánh giá:")
        print("   - Việc bỏ sót một máy đang quá tải (False Negative) cực kỳ nguy hiểm, có thể dẫn đến cháy nổ.")
        print("   - Báo động nhầm (False Positive) an toàn hơn, chỉ làm tốn một chút thời gian dừng máy kiểm tra.")
    else:
        print("[!] Không thể phân tích vì ma trận không phải 2x2. Có lỗi bất ngờ xảy ra.")
    print("====================================\n")

if __name__ == "__main__":
    main()