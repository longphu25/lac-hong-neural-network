from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# BÀI TẬP PERCEPTRON – QUYẾT ĐỊNH TƯỚI CÂY
# ============================================================
# Nhãn:
#   Relay = 0 -> Chưa cần tưới
#   Relay = 1 -> Cần tưới
#
# Feature:
#   Moist              : độ ẩm đất
#   Temperature        : nhiệt độ không khí
#   HoursSinceWatering : số giờ từ lần tưới gần nhất
#   RainChance         : khả năng mưa
#
# Không có feature phân loại nên chỉ cần StandardScaler.
# ============================================================
from pathlib import Path
import pandas as pd

from pathlib import Path
import pandas as pd

# Thư mục gốc lac-hong-neural-network
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Dataset nằm trong thư mục data
CSV_FILE = BASE_DIR / "data" / "du_lieu_tuoi_cay_perceptron.csv"

print("Đang đọc dataset:")
print(CSV_FILE)

if not CSV_FILE.exists():
    raise FileNotFoundError(
        f"Không tìm thấy dataset tại:\n{CSV_FILE}"
    )

df = pd.read_csv(CSV_FILE)

print("Đọc dữ liệu thành công!")
print(df.head())

FEATURE_COLUMNS = [
    "Moist",
    "Temperature",
    "HoursSinceWatering",
    "RainChance",
]

TARGET_COLUMN = "Relay"

CLASS_NAMES = {
    0: "Chưa cần tưới",
    1: "Cần tưới",
}


def load_data():
    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file CSV tại:\n{CSV_FILE}\n"
            "Hãy đặt file CSV cùng thư mục với file Python."
        )

    data = pd.read_csv(CSV_FILE)

    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(
            "CSV thiếu các cột: " + ", ".join(missing)
        )

    return data


def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "perceptron",
            Perceptron(
                max_iter=1000,
                tol=1e-3,
                eta0=0.1,
                random_state=42,
                shuffle=True,
            ),
        ),
    ])


def train_and_evaluate(data):
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    net = model.decision_function(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    print("=" * 65)
    print("PERCEPTRON – QUYẾT ĐỊNH TƯỚI CÂY")
    print("=" * 65)

    print(f"Số mẫu: {len(data)}")
    print(f"Train: {len(X_train)}")
    print(f"Test : {len(X_test)}")

    print("\nPhân bố nhãn:")
    print(
        data[TARGET_COLUMN]
        .map(CLASS_NAMES)
        .value_counts()
    )

    print("\nFeature sử dụng:")
    for feature in FEATURE_COLUMNS:
        print(" -", feature)

    print("\nAccuracy:", round(accuracy, 4))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=[
                CLASS_NAMES[0],
                CLASS_NAMES[1],
            ],
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(cm)

    perceptron = model.named_steps["perceptron"]

    print("\nBias b:", round(float(perceptron.intercept_[0]), 6))
    print("Số vòng lặp:", perceptron.n_iter_)

    print("\nTrọng số sau StandardScaler:")
    for feature, weight in zip(
        FEATURE_COLUMNS,
        perceptron.coef_[0],
    ):
        print(f"  {feature}: {weight:.6f}")

    print("\n5 giá trị net đầu tiên:")
    print(np.round(net[:5], 6))

    # Kết quả tập test
    test_result = data.loc[X_test.index].copy()
    test_result["nhan_that"] = y_test
    test_result["du_doan"] = y_pred
    test_result["net"] = net
    test_result["ket_qua"] = np.where(
        test_result["nhan_that"] == test_result["du_doan"],
        "Đúng",
        "Sai",
    )

    output_test = BASE_DIR / "ket_qua_test_tuoi_cay.csv"
    test_result.to_csv(
        output_test,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nĐã lưu:", output_test.name)

    return model, test_result


def predict_new_plant(model, sample):
    new_data = pd.DataFrame(
        [sample],
        columns=FEATURE_COLUMNS,
    )

    prediction = int(model.predict(new_data)[0])
    net = float(model.decision_function(new_data)[0])

    print("\n" + "=" * 65)
    print("DỰ ĐOÁN NHÀ MỚI / TÌNH TRẠNG CÂY")
    print("=" * 65)

    for feature in FEATURE_COLUMNS:
        print(f"{feature}: {sample[feature]}")

    print("\nnet =", round(net, 6))
    print("Relay =", prediction)
    print("Quyết định =", CLASS_NAMES[prediction])

    return prediction


def main():
    data = load_data()

    print("Đọc dữ liệu thành công!")
    print("Kích thước:", data.shape)
    print("\n5 dòng đầu:")
    print(data.head().to_string(index=False))

    model, test_result = train_and_evaluate(data)

    # Mẫu dự đoán thử.
    sample = {
        "Moist": 35,
        "Temperature": 32,
        "HoursSinceWatering": 8,
        "RainChance": 15,
    }

    predict_new_plant(model, sample)


if __name__ == "__main__":
    main()
