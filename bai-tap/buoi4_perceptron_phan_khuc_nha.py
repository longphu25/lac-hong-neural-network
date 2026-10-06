"""Buổi 4 — Perceptron phân loại phân khúc nhà từ dữ liệu thực tế mô phỏng.

Bài toán: dự đoán một căn nhà thuộc nhóm giá/m² cao hay phổ thông dựa trên
vị trí, loại nhà và đặc điểm vật lý. Hai cột giá chỉ được dùng để tạo nhãn,
không được đưa vào feature vì sẽ làm rò rỉ đáp án (data leakage).

Chạy từ thư mục gốc repo:
    python bai-tap/buoi4_perceptron_phan_khuc_nha.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Perceptron
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_houses.csv"
DEFAULT_THRESHOLD = 85.0  # triệu đồng/m²
TARGET_COLUMN = "phan_khuc_cao"
TARGET_SOURCE_COLUMN = "price_per_m2_million_vnd"

NUMERIC_FEATURES = [
    "area_m2",
    "bedrooms",
    "bathrooms",
    "floors",
    "year_built",
]
CATEGORICAL_FEATURES = ["city", "district", "building_type"]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
LEAKAGE_COLUMNS = ["price_million_vnd", TARGET_SOURCE_COLUMN]
CLASS_NAMES = {0: "Phổ thông", 1: "Giá/m² cao"}


@dataclass
class ExperimentResult:
    """Kết quả cần thiết để đánh giá và tiếp tục thử nghiệm."""

    model: Pipeline
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    predictions: np.ndarray
    metrics: dict[str, float]
    matrix: np.ndarray
    report: pd.DataFrame


def load_housing_data(
    data_path: str | Path = DEFAULT_DATA_PATH,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Đọc CSV, kiểm tra cột và tạo nhãn nhị phân từ ngưỡng giá/m²."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu: {path}")

    data = pd.read_csv(path)
    required = set(FEATURE_COLUMNS + LEAKAGE_COLUMNS)
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"CSV đang thiếu các cột bắt buộc: {missing}")
    if data[list(required)].isna().any().any():
        raise ValueError("Dữ liệu có giá trị trống; hãy xử lý trước khi huấn luyện.")
    if threshold <= 0:
        raise ValueError("Ngưỡng giá/m² phải lớn hơn 0.")

    prepared = data.copy()
    prepared[TARGET_COLUMN] = (
        prepared[TARGET_SOURCE_COLUMN] >= threshold
    ).astype(int)
    return prepared


def split_dataset(
    data: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Chia train/test có giữ tỷ lệ hai lớp."""
    if not 0 < test_size < 1:
        raise ValueError("test_size phải nằm trong khoảng (0, 1).")

    x = data[FEATURE_COLUMNS].copy()
    y = data[TARGET_COLUMN].copy()
    if y.nunique() < 2:
        raise ValueError("Ngưỡng đã tạo ra ít hơn hai lớp; hãy chọn ngưỡng khác.")

    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_model(random_state: int = 42, eta0: float = 0.1) -> Pipeline:
    """Tạo pipeline tiền xử lý và một node Perceptron nhị phân."""
    if eta0 <= 0:
        raise ValueError("eta0 phải lớn hơn 0.")

    preprocess = ColumnTransformer(
        transformers=[
            ("number", StandardScaler(), NUMERIC_FEATURES),
            (
                "category",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    perceptron = Perceptron(
        eta0=eta0,
        max_iter=2_000,
        tol=1e-4,
        random_state=random_state,
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("perceptron", perceptron),
        ]
    )


def run_experiment(
    data_path: str | Path = DEFAULT_DATA_PATH,
    threshold: float = DEFAULT_THRESHOLD,
    test_size: float = 0.25,
    random_state: int = 42,
    eta0: float = 0.1,
) -> tuple[pd.DataFrame, ExperimentResult]:
    """Nạp dữ liệu, chia tập, huấn luyện và đánh giá mô hình."""
    data = load_housing_data(data_path, threshold)
    x_train, x_test, y_train, y_test = split_dataset(
        data,
        test_size=test_size,
        random_state=random_state,
    )
    model = build_model(random_state=random_state, eta0=eta0)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision_lop_cao": precision_score(y_test, predictions, zero_division=0),
        "recall_lop_cao": recall_score(y_test, predictions, zero_division=0),
        "f1_lop_cao": f1_score(y_test, predictions, zero_division=0),
    }
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    report = pd.DataFrame(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=[CLASS_NAMES[0], CLASS_NAMES[1]],
            output_dict=True,
            zero_division=0,
        )
    ).transpose()

    result = ExperimentResult(
        model=model,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        predictions=predictions,
        metrics=metrics,
        matrix=matrix,
        report=report,
    )
    return data, result


def feature_weight_table(model: Pipeline) -> pd.DataFrame:
    """Lấy trọng số sau tiền xử lý để nối lại với công thức net = wᵀx + b."""
    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    weights = model.named_steps["perceptron"].coef_[0]
    return pd.DataFrame({"feature": feature_names, "weight": weights}).sort_values(
        "weight",
        ascending=False,
    )


def predict_new_houses(model: Pipeline, houses: list[dict[str, Any]]) -> pd.DataFrame:
    """Dự đoán các căn nhà mới và trả cả net lẫn nhãn dễ đọc."""
    frame = pd.DataFrame(houses)
    missing = sorted(set(FEATURE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Mẫu mới đang thiếu feature: {missing}")

    x_new = frame[FEATURE_COLUMNS]
    predicted = model.predict(x_new).astype(int)
    net_scores = model.decision_function(x_new)
    output = frame.copy()
    output["net"] = np.round(net_scores, 3)
    output["du_doan"] = [CLASS_NAMES[label] for label in predicted]
    return output


def print_summary(data: pd.DataFrame, result: ExperimentResult, threshold: float) -> None:
    """In báo cáo ngắn, đủ để sinh viên kiểm tra một lần chạy."""
    counts = data[TARGET_COLUMN].value_counts().sort_index()
    print("\n=== BÀI TOÁN ===")
    print(
        f"Dự đoán 'Giá/m² cao' khi giá/m² >= {threshold:.1f} triệu đồng; "
        "giá không được dùng làm input."
    )
    print(f"Số mẫu: {len(data)} | train: {len(result.x_train)} | test: {len(result.x_test)}")
    print(
        "Phân bố lớp: "
        + ", ".join(f"{CLASS_NAMES[int(label)]}={count}" for label, count in counts.items())
    )
    print(f"Feature: {', '.join(FEATURE_COLUMNS)}")
    print(f"Loại để tránh leakage: {', '.join(LEAKAGE_COLUMNS)}")

    print("\n=== KẾT QUẢ TRÊN TẬP TEST ===")
    for name, value in result.metrics.items():
        print(f"{name}: {value:.3f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(result.matrix)
    print("\nClassification report:")
    print(result.report.round(3).to_string())

    weights = feature_weight_table(result.model)
    print("\n5 tín hiệu kéo dự đoán về lớp 'Giá/m² cao':")
    print(weights.head(5).to_string(index=False))
    print("\n5 tín hiệu kéo dự đoán về lớp 'Phổ thông':")
    print(weights.tail(5).sort_values("weight").to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Huấn luyện Perceptron phân loại phân khúc nhà."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eta", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data, result = run_experiment(
        data_path=args.data,
        threshold=args.threshold,
        test_size=args.test_size,
        random_state=args.seed,
        eta0=args.eta,
    )
    print_summary(data, result, args.threshold)

    sample_houses = [
        {
            "city": "Ho Chi Minh City",
            "district": "Thu Duc",
            "building_type": "apartment",
            "area_m2": 70.0,
            "bedrooms": 2,
            "bathrooms": 2,
            "floors": 1,
            "year_built": 2020,
        },
        {
            "city": "Ha Noi",
            "district": "Hoan Kiem",
            "building_type": "townhouse",
            "area_m2": 75.0,
            "bedrooms": 3,
            "bathrooms": 2,
            "floors": 4,
            "year_built": 2018,
        },
    ]
    print("\n=== CHẠY THỬ HAI CĂN NHÀ MỚI ===")
    print(
        predict_new_houses(result.model, sample_houses)[
            ["city", "district", "building_type", "area_m2", "net", "du_doan"]
        ].to_string(index=False)
    )
    print("\nLưu ý: dữ liệu mô phỏng chỉ dùng cho học tập, không phải tư vấn định giá.")


if __name__ == "__main__":
    main()
