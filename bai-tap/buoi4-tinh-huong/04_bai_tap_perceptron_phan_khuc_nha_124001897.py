from pathlib import Path
import pandas as pd
import numpy as np

base = Path("data")
csv_path = base / "Housing.csv"

# Đọc dữ liệu gốc
df = pd.read_csv(csv_path)

# ============================================================
# PHÂN KHÚC NHÀ + PERCEPTRON
# Dựa trên dàn ý notebook người dùng cung cấp, nhưng điều chỉnh
# tên feature để khớp trực tiếp với Housing.csv thực tế.
#
# Lưu ý:
# - Housing.csv có price và area, nhưng không có đơn vị VND/m².
# - price/area trong file này nằm khoảng 270 -> 2640.
# - Vì vậy dùng ngưỡng 850 trên price/area để tạo 2 lớp cân bằng
#   tương đối. Nếu giảng viên yêu cầu đúng ngưỡng 85, đổi THRESHOLD.
# - price và price_per_area KHÔNG được đưa vào feature để tránh leakage.
# ============================================================

script = r'''from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# =========================
# 1. CẤU HÌNH
# =========================
BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "Housing.csv"

# Housing.csv không ghi đơn vị của price/area.
# price/area có median khoảng 952.38, nên 850 tạo được 2 lớp.
THRESHOLD = 850.0

TARGET_COLUMN = "segment"
TARGET_SOURCE_COLUMN = "price_per_area"

CLASS_NAMES = {
    0: "Phổ thông",
    1: "Giá/mật độ giá cao",
}

# Không dùng price hoặc price_per_area làm feature vì chúng trực tiếp
# tạo ra nhãn.
LEAKAGE_COLUMNS = ["price", "price_per_area", "segment"]

NUMERIC_FEATURES = [
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "parking",
]

CATEGORICAL_FEATURES = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# =========================
# 2. ĐỌC CSV + TẠO NHÃN
# =========================
def load_housing_data(csv_file=CSV_FILE, threshold=THRESHOLD):
    data = pd.read_csv(csv_file)

    required = ["price", "area"] + FEATURE_COLUMNS
    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(
            "Thiếu cột trong Housing.csv: " + ", ".join(missing)
        )

    if (data["area"] <= 0).any():
        raise ValueError("Cột area phải lớn hơn 0.")

    # Chỉ dùng để tạo nhãn, không dùng làm đầu vào.
    data["price_per_area"] = data["price"] / data["area"]

    data[TARGET_COLUMN] = (
        data["price_per_area"] >= threshold
    ).astype(int)

    return data


# =========================
# 3. TẠO MODEL PERCEPTRON
# =========================
def build_model(random_state=42, eta0=0.1):
    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    perceptron = Perceptron(
        max_iter=1000,
        tol=1e-3,
        eta0=eta0,
        random_state=random_state,
        shuffle=True,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocess),
            ("perceptron", perceptron),
        ]
    )

    return model


# =========================
# 4. CHIA TRAIN / TEST
# =========================
def split_dataset(data, test_size=0.25, random_state=42):
    X = data[FEATURE_COLUMNS].copy()
    y = data[TARGET_COLUMN].copy()

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


# =========================
# 5. CHẠY HUẤN LUYỆN
# =========================
def train_and_evaluate(data):
    X_train, X_test, y_train, y_test = split_dataset(data)

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    net_scores = model.decision_function(X_test)

    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])

    print("=" * 65)
    print("PERCEPTRON PHÂN KHÚC NHÀ")
    print("=" * 65)

    print(f"Số mẫu: {len(data)}")
    print(f"Ngưỡng price/area: {THRESHOLD}")
    print("\nPhân bố nhãn:")
    print(
        data[TARGET_COLUMN]
        .map(CLASS_NAMES)
        .value_counts()
    )

    print("\nFeature sử dụng:")
    for feature in FEATURE_COLUMNS:
        print(" -", feature)

    print("\nTrain:", X_train.shape)
    print("Test :", X_test.shape)

    print("\nAccuracy:", round(accuracy, 4))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=[
                CLASS_NAMES[0],
                CLASS_NAMES[1],
            ],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(cm)

    perceptron = model.named_steps["perceptron"]

    print("\nBias b:", round(float(perceptron.intercept_[0]), 4))
    print("Số vòng lặp:", perceptron.n_iter_)

    print("\n5 giá trị net đầu tiên:")
    print(np.round(net_scores[:5], 4))

    # Tạo bảng test có thông tin gốc
    test_result = data.loc[X_test.index].copy()
    test_result["nhan_that"] = y_test
    test_result["du_doan"] = predictions
    test_result["net"] = net_scores
    test_result["ket_qua"] = np.where(
        test_result["nhan_that"] == test_result["du_doan"],
        "Đúng",
        "Sai",
    )

    sai = test_result[
        test_result["nhan_that"] != test_result["du_doan"]
    ]

    print(f"\nSố mẫu dự đoán sai: {len(sai)}")

    # Lưu kết quả test
    test_output = BASE_DIR / "ket_qua_test_phan_khuc_nha.csv"
    test_result.to_csv(test_output, index=False, encoding="utf-8-sig")
    print("Đã lưu:", test_output.name)

    return model, test_result


# =========================
# 6. DỰ ĐOÁN NHÀ MỚI
# =========================
def predict_new_house(model, house):
    new_data = pd.DataFrame([house])[FEATURE_COLUMNS]

    prediction = int(model.predict(new_data)[0])
    net = float(model.decision_function(new_data)[0])

    print("\n" + "=" * 65)
    print("DỰ ĐOÁN NHÀ MỚI")
    print("=" * 65)

    for key, value in house.items():
        print(f"{key}: {value}")

    print("\nnet =", round(net, 4))
    print("Nhãn =", prediction)
    print("Phân khúc =", CLASS_NAMES[prediction])

    return prediction


# =========================
# 7. MAIN
# =========================
def main():
    data = load_housing_data()

    # Xuất file phân khúc toàn bộ dữ liệu.
    segmented = data.copy()
    segmented["phan_khuc"] = segmented[TARGET_COLUMN].map(CLASS_NAMES)

    output_file = BASE_DIR / "phan_khuc_nha.csv"
    segmented.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    print("Đã tạo file:", output_file.name)
    print(
        "\nTổng số nhà:",
        len(segmented),
        "| Phổ thông:",
        int((segmented[TARGET_COLUMN] == 0).sum()),
        "| Giá cao:",
        int((segmented[TARGET_COLUMN] == 1).sum()),
    )

    # Train + evaluate
    model, test_result = train_and_evaluate(data)

    # Nhà mẫu để chạy thử.
    sample_house = {
        "area": 5000,
        "bedrooms": 3,
        "bathrooms": 2,
        "stories": 2,
        "parking": 1,
        "mainroad": "yes",
        "guestroom": "no",
        "basement": "yes",
        "hotwaterheating": "no",
        "airconditioning": "yes",
        "prefarea": "yes",
        "furnishingstatus": "semi-furnished",
    }

    predict_new_house(model, sample_house)


if __name__ == "__main__":
    main()
'''

py_path = base / "phan_khuc_nha_perceptron.py"
py_path.write_text(script, encoding="utf-8")

# Chạy chính xác logic tương đương để tạo file phân khúc từ CSV.
THRESHOLD = 850.0
segmented = df.copy()
segmented["price_per_area"] = segmented["price"] / segmented["area"]
segmented["segment"] = (segmented["price_per_area"] >= THRESHOLD).astype(int)
segmented["phan_khuc"] = segmented["segment"].map({
    0: "Phổ thông",
    1: "Giá/mật độ giá cao",
})
seg_path = base / "phan_khuc_nha.csv"
segmented.to_csv(seg_path, index=False, encoding="utf-8-sig")

# Kiểm tra nhanh dữ liệu đầu ra.
print(f"Đã tạo: {py_path}")
print(f"Đã tạo: {seg_path}")
print("Số dòng:", len(segmented))
print(segmented["phan_khuc"].value_counts().to_string())

# Test import/compile file Python.
compile(script, str(py_path), "exec")
print("Kiểm tra cú pháp Python: OK")
