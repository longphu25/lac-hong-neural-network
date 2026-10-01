import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix

def main():
    # 1. Tải dữ liệu
    data_path = '../data/industrial_fault_detection_data_1000.csv'
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file tại {data_path}.")
        return

    # 2. Tiền xử lý dữ liệu
    df_numeric = df.select_dtypes(include=[np.number])
    X = df_numeric.iloc[:, :-1].values
    y_raw = df_numeric.iloc[:, -1].values
    
    # Đưa về bài toán phân loại nhị phân (0 = Bình thường, 1 = Quá tải)
    y = np.where(y_raw > 0, 1, 0)

    # 3. Chuẩn hoá dữ liệu
    sc = StandardScaler()
    X_std = sc.fit_transform(X)

    # 4. Huấn luyện Perceptron (Sử dụng class_weight='balanced' để ép mô hình chú ý vào nhãn 1)
    print("⏳ Đang huấn luyện mô hình Perceptron...")
    ppn = Perceptron(eta0=0.1, max_iter=1000, class_weight='balanced', random_state=42)
    ppn.fit(X_std, y)

    # 5. Đánh giá và Phân tích
    y_pred = ppn.predict(X_std)
    print(f"🎯 Độ chính xác (Accuracy): {accuracy_score(y, y_pred) * 100:.2f}%\n")

    cm = confusion_matrix(y, y_pred)
    print("📊 Ma trận nhầm lẫn (Confusion Matrix):")
    print(cm)
    
    false_negative = cm[1, 0] 
    false_positive = cm[0, 1] 
    
    print("\n--- PHÂN TÍCH ---")
    print(f"Bỏ sót máy quá tải (False Negative): {false_negative} trường hợp.")
    print(f"Báo động nhầm (False Positive): {false_positive} trường hợp.")

if __name__ == '__main__':
    main()