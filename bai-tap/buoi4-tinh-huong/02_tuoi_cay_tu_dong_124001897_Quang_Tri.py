# ============================================================
# BUỔI 5 - TÌNH HUỐNG 02: QUYẾT ĐỊNH TƯỚI CÂY (ADALINE)
# Sinh viên: 124001897 - Dương Quang Trí
#
# Mục tiêu:
# - Đọc dữ liệu TARP.csv
# - Dùng Adaline (Widrow-Hoff / LMS) để học nhu cầu tưới
# - Dự đoán giá trị liên tục trong khoảng 0..1
# - Chuyển giá trị dự đoán thành:
#       0 = chưa cần tưới
#       1 = cần tưới
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. CẤU HÌNH
# ------------------------------------------------------------

# Nếu TARP.csv nằm cùng thư mục với file Python:
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "TARP.csv")

# Các feature lấy từ dữ liệu TARP phù hợp với bài toán tưới cây.
# Soil Moisture : độ ẩm đất (%)
# Temperature   : nhiệt độ không khí
# Time          : thời gian
# rainfall      : lượng mưa
FEATURES = [
    "Soil Moisture",
    "Temperature",
    "Time",
    "rainfall"
]

TARGET_COLUMN = "Status"

# Adaline
LEARNING_RATE = 0.001
EPOCHS = 100
THRESHOLD = 0.50

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ------------------------------------------------------------
# 2. ĐỌC DỮ LIỆU
# ------------------------------------------------------------

def load_data():
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(
            f"Không tìm thấy TARP.csv tại:\n{CSV_FILE}\n"
            "Hãy đặt TARP.csv cùng thư mục với file Python."
        )

    df = pd.read_csv(CSV_FILE)

    # Xóa khoảng trắng thừa ở tên cột nếu có
    df.columns = df.columns.str.strip()

    required_columns = FEATURES + [TARGET_COLUMN]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            "CSV thiếu các cột:\n" + "\n".join(missing)
        )

    # Chỉ lấy các cột cần thiết
    data = df[required_columns].copy()

    # Chuyển các feature về số
    for col in FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Loại bỏ dòng bị thiếu dữ liệu
    data = data.dropna().reset_index(drop=True)

    # Status:
    # ON  -> 1: cần tưới
    # OFF -> 0: chưa cần tưới
    data[TARGET_COLUMN] = (
        data[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .str.upper()
        .map({"ON": 1.0, "OFF": 0.0})
    )

    data = data.dropna().reset_index(drop=True)

    return data


# ------------------------------------------------------------
# 3. CHIA TRAIN / TEST
# ------------------------------------------------------------

def train_test_split(X, y, test_size=0.2, random_state=42):
    rng = np.random.default_rng(random_state)

    indices = np.arange(len(X))
    rng.shuffle(indices)

    test_count = int(len(X) * test_size)

    test_idx = indices[:test_count]
    train_idx = indices[test_count:]

    return (
        X[train_idx],
        X[test_idx],
        y[train_idx],
        y[test_idx]
    )


# ------------------------------------------------------------
# 4. CHUẨN HÓA MIN-MAX
# ------------------------------------------------------------

def fit_minmax(X):
    min_value = X.min(axis=0)
    max_value = X.max(axis=0)

    # Tránh chia cho 0 nếu một feature có cùng một giá trị
    scale = max_value - min_value
    scale[scale == 0] = 1.0

    return min_value, scale


def transform_minmax(X, min_value, scale):
    return (X - min_value) / scale


# ------------------------------------------------------------
# 5. ADALINE
# ------------------------------------------------------------

class Adaline:
    """
    Adaline sử dụng quy tắc Widrow-Hoff / LMS:

        net = w.x + b
        error = target - net

        w = w + eta * error * x
        b = b + eta * error

    Target ở đây là 0 hoặc 1.
    Tuy nhiên Adaline học giá trị thực liên tục,
    sau đó ta dùng threshold để đưa về quyết định ON/OFF.
    """

    def __init__(self, learning_rate=0.001, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.weights = None
        self.bias = 0.0
        self.mse_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0
        self.mse_history = []

        for epoch in range(self.epochs):
            errors = []

            # LMS: cập nhật sau từng mẫu
            for xi, target in zip(X, y):

                # net = w.x + b
                output = np.dot(xi, self.weights) + self.bias

                # e = t - net
                error = target - output

                # Widrow-Hoff
                self.weights += (
                    self.learning_rate * error * xi
                )

                self.bias += self.learning_rate * error

                errors.append(error ** 2)

            mse = np.mean(errors)
            self.mse_history.append(mse)

        return self

    def net_input(self, X):
        return np.dot(X, self.weights) + self.bias

    def predict_continuous(self, X):
        """
        Trả về giá trị liên tục.
        Giá trị càng gần 1 -> nhu cầu tưới càng cao.
        """
        return self.net_input(X)

    def predict(self, X, threshold=0.5):
        """
        Chuyển kết quả liên tục thành:
        0 = OFF = chưa cần tưới
        1 = ON  = cần tưới
        """
        continuous = self.predict_continuous(X)
        return (continuous >= threshold).astype(int)


# ------------------------------------------------------------
# 6. ĐÁNH GIÁ
# ------------------------------------------------------------

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# ------------------------------------------------------------
# 7. DỰ ĐOÁN MỘT MẪU MỚI
# ------------------------------------------------------------

def predict_new_sample(model, min_value, scale):
    print("\n" + "=" * 65)
    print("DỰ ĐOÁN HỆ THỐNG TƯỚI CÂY")
    print("=" * 65)

    print("\nNhập thông số cảm biến:")

    soil_moisture = float(
        input("Độ ẩm đất (Soil Moisture): ")
    )

    temperature = float(
        input("Nhiệt độ không khí (Temperature): ")
    )

    time_value = float(
        input("Thời gian (Time): ")
    )

    rainfall = float(
        input("Lượng mưa (rainfall): ")
    )

    sample = np.array(
        [[
            soil_moisture,
            temperature,
            time_value,
            rainfall
        ]],
        dtype=float
    )

    # Chuẩn hóa giống dữ liệu train
    sample_scaled = transform_minmax(
        sample,
        min_value,
        scale
    )

    # Adaline trả về giá trị liên tục
    score = model.predict_continuous(sample_scaled)[0]

    # Giới hạn điểm trong khoảng 0..1 để dễ giải thích
    score_clipped = np.clip(score, 0, 1)

    decision = 1 if score_clipped >= THRESHOLD else 0

    print("\nKẾT QUẢ:")
    print(f"Mức độ cần tưới: {score_clipped:.4f}")
    print(f"Ngưỡng quyết định: {THRESHOLD}")

    if decision == 1:
        print("=> BẬT MÁY BƠM: CẦN TƯỚI CÂY")
    else:
        print("=> TẮT MÁY BƠM: CHƯA CẦN TƯỚI CÂY")

    return score_clipped, decision


# ------------------------------------------------------------
# 8. CHƯƠNG TRÌNH CHÍNH
# ------------------------------------------------------------

def main():

    print("=" * 65)
    print("ADALINE - HỆ THỐNG TƯỚI CÂY TỰ ĐỘNG")
    print("=" * 65)

    # Đọc dữ liệu
    df = load_data()

    print(f"\nSố mẫu dữ liệu: {len(df)}")
    print(f"Feature sử dụng: {FEATURES}")
    print(f"Target: {TARGET_COLUMN}")
    print("ON = 1 (cần tưới), OFF = 0 (chưa cần tưới)")

    # --------------------------------------------------------
    # X và y
    # --------------------------------------------------------

    X = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET_COLUMN].to_numpy(dtype=float)

    # --------------------------------------------------------
    # Chia train/test
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print(f"\nTrain: {len(X_train)} mẫu")
    print(f"Test : {len(X_test)} mẫu")

    # --------------------------------------------------------
    # Chuẩn hóa
    # --------------------------------------------------------

    min_value, scale = fit_minmax(X_train)

    X_train_scaled = transform_minmax(
        X_train,
        min_value,
        scale
    )

    X_test_scaled = transform_minmax(
        X_test,
        min_value,
        scale
    )

    # --------------------------------------------------------
    # DỰ ĐOÁN BẰNG TAY TRƯỚC KHI TRAIN
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("DỰ ĐOÁN THỬ BẰNG CÔNG THỨC ADALINE")
    print("=" * 65)

    # Trọng số ban đầu bằng 0 nên net ban đầu = 0
    sample_before = X_train_scaled[0]

    initial_w = np.zeros(len(FEATURES))
    initial_b = 0.0

    initial_net = (
        np.dot(initial_w, sample_before)
        + initial_b
    )

    print("Mẫu đầu tiên sau chuẩn hóa:")
    print(sample_before)
    print(f"net ban đầu = {initial_net:.4f}")
    print(
        "Sau đó Adaline sẽ cập nhật w và b "
        "theo quy tắc Widrow-Hoff."
    )

    # --------------------------------------------------------
    # TRAIN ADALINE
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("HUẤN LUYỆN ADALINE")
    print("=" * 65)

    model = Adaline(
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS
    )

    model.fit(X_train_scaled, y_train)

    print("\nTrọng số sau huấn luyện:")

    for feature, weight in zip(
        FEATURES,
        model.weights
    ):
        print(f"{feature:25s}: {weight:.6f}")

    print(f"Bias                    : {model.bias:.6f}")

    # --------------------------------------------------------
    # ĐÁNH GIÁ TRAIN
    # --------------------------------------------------------

    train_score = model.predict_continuous(
        X_train_scaled
    )

    train_pred = model.predict(
        X_train_scaled,
        THRESHOLD
    )

    # --------------------------------------------------------
    # ĐÁNH GIÁ TEST
    # --------------------------------------------------------

    test_score = model.predict_continuous(
        X_test_scaled
    )

    test_pred = model.predict(
        X_test_scaled,
        THRESHOLD
    )

    train_acc = accuracy(y_train, train_pred)
    test_acc = accuracy(y_test, test_pred)

    train_mse = mse(y_train, train_score)
    test_mse = mse(y_test, test_score)

    print("\n" + "=" * 65)
    print("KẾT QUẢ ĐÁNH GIÁ")
    print("=" * 65)

    print(f"Train accuracy: {train_acc:.2f}%")
    print(f"Test accuracy : {test_acc:.2f}%")
    print(f"Train MSE     : {train_mse:.6f}")
    print(f"Test MSE      : {test_mse:.6f}")

    # --------------------------------------------------------
    # HIỂN THỊ MỘT SỐ DỰ ĐOÁN
    # --------------------------------------------------------

    result = pd.DataFrame({
        "Thực tế": y_test[:20].astype(int),
        "Điểm Adaline": np.round(
            np.clip(test_score[:20], 0, 1),
            4
        ),
        "Dự đoán": test_pred[:20]
    })

    print("\n20 mẫu test đầu tiên:")
    print(result.to_string(index=False))

    # --------------------------------------------------------
    # VẼ MSE
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, EPOCHS + 1),
        model.mse_history
    )
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Quá trình học của Adaline")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # NHẬN XÉT TRỌNG SỐ
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("PHÂN TÍCH TRỌNG SỐ")
    print("=" * 65)

    for feature, weight in zip(
        FEATURES,
        model.weights
    ):
        if weight > 0:
            direction = "tăng nhu cầu tưới"
        elif weight < 0:
            direction = "giảm nhu cầu tưới"
        else:
            direction = "ảnh hưởng gần như bằng 0"

        print(
            f"- {feature}: {weight:.6f} "
            f"-> {direction}"
        )

    print("\nLưu ý:")
    print(
        "Dấu của trọng số cần được đọc sau khi chuẩn hóa "
        "và nên đối chiếu với dữ liệu thực tế."
    )

    # --------------------------------------------------------
    # DỰ ĐOÁN MẪU MỚI
    # --------------------------------------------------------

    answer = input(
        "\nBạn có muốn nhập dữ liệu cảm biến mới? (y/n): "
    ).strip().lower()

    if answer == "y":
        predict_new_sample(
            model,
            min_value,
            scale
        )

    print("\n" + "=" * 65)
    print("HOÀN THÀNH")
    print("=" * 65)


if __name__ == "__main__":
    main()
