"""Buổi 4 — Perceptron cải tiến: Phân loại cảnh báo rủi ro / nhập kho."""

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
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed_cleveland.csv"
DEFAULT_THRESHOLD = 2.0
TARGET_COLUMN = "canh_bao_rui_ro"
TARGET_SOURCE_COLUMN = "target"

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
LEAKAGE_COLUMNS = [TARGET_SOURCE_COLUMN]
CLASS_NAMES = {0: "An toàn / Ổn định", 1: "Cảnh báo rủi ro cao"}

@dataclass
class ExperimentResult:
    model: Pipeline
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    predictions: np.ndarray
    metrics: dict[str, float]
    matrix: np.ndarray
    report: pd.DataFrame
    n_iter: int

def load_stockout_data(
    data_path: str | Path = DEFAULT_DATA_PATH,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu: {path}")

    column_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", TARGET_SOURCE_COLUMN
    ]
    
    data = pd.read_csv(path, header=None, names=column_names, na_values="?")
    data = data.dropna()

    required = set(FEATURE_COLUMNS + LEAKAGE_COLUMNS)
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Dữ liệu đang thiếu các cột bắt buộc: {missing}")

    prepared = data.copy()
    for col in FEATURE_COLUMNS + [TARGET_SOURCE_COLUMN]:
        prepared[col] = pd.to_numeric(prepared[col], errors='coerce')
    
    prepared = prepared.dropna()
    prepared[TARGET_COLUMN] = (prepared[TARGET_SOURCE_COLUMN] >= threshold).astype(int)
    return prepared

def split_dataset(
    data: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    x = data[FEATURE_COLUMNS].copy()
    y = data[TARGET_COLUMN].copy()
    if y.nunique() < 2:
        raise ValueError("Ngưỡng đã tạo ra ít hơn hai lớp; hãy chọn ngưỡng khác.")

    return train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)

def build_model(random_state: int = 42, eta0: float = 0.1, use_scaler: bool = True) -> Pipeline:
    transformers = [
        ("category", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ]
    if use_scaler:
        transformers.append(("number", StandardScaler(), NUMERIC_FEATURES))
    else:
        transformers.append(("number", FunctionTransformer(validate=False), NUMERIC_FEATURES))

    preprocess = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    perceptron = Perceptron(
        eta0=eta0, max_iter=2_000, tol=1e-4, random_state=random_state
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
    use_scaler: bool = True,
) -> tuple[pd.DataFrame, ExperimentResult]:
    data = load_stockout_data(data_path, threshold)
    x_train, x_test, y_train, y_test = split_dataset(
        data, test_size=test_size, random_state=random_state
    )
    model = build_model(random_state=random_state, eta0=eta0, use_scaler=use_scaler)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    n_iter = int(model.named_steps["perceptron"].n_iter_)
    
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision_lop_cao": precision_score(y_test, predictions, zero_division=0),
        "recall_lop_cao": recall_score(y_test, predictions, zero_division=0),
        "f1_lop_cao": f1_score(y_test, predictions, zero_division=0),
    }
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    report = pd.DataFrame(
        classification_report(
            y_test, predictions, labels=[0, 1],
            target_names=[CLASS_NAMES[0], CLASS_NAMES[1]],
            output_dict=True, zero_division=0,
        )
    ).transpose()

    result = ExperimentResult(
        model=model, x_train=x_train, x_test=x_test,
        y_train=y_train, y_test=y_test, predictions=predictions,
        metrics=metrics, matrix=matrix, report=report, n_iter=n_iter,
    )
    return data, result

def feature_weight_table(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocess"]
    perceptron = model.named_steps["perceptron"]
    
    feature_names = preprocessor.get_feature_names_out()
    weights = perceptron.coef_[0]
    
    df = pd.DataFrame({
        "feature": feature_names,
        "weight": weights
    })
    return df.sort_values("weight", ascending=False).reset_index(drop=True)

def print_summary(data: pd.DataFrame, result: ExperimentResult, threshold: float) -> None:
    print(f"\n--- KẾT QUẢ VỚI NGƯỠNG = {threshold} ---")
    print(f"Số mẫu hợp lệ: {len(data)} | Số vòng lặp hội tụ (n_iter): {result.n_iter}")
    for name, value in result.metrics.items():
        print(f"  {name}: {value:.3f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(result.matrix)

def main() -> None:
    print("=== 1. KHẢO SÁT CÁC NGƯỠNG PHÂN LOẠI KHÁC NHAU ===")
    for th in [1.5, 2.0, 2.5]:
        data, result = run_experiment(threshold=th)
        print_summary(data, result, th)

    print("\n=== 2. SO SÁNH ẢNH HƯỞNG CỦA CHUẨN HÓA DỮ LIỆU (StandardScaler) ===")
    print(">> Có dùng StandardScaler (Chuẩn hóa):")
    _, res_scaled = run_experiment(threshold=2.0, use_scaler=True)
    print(f"   Accuracy: {res_scaled.metrics['accuracy']:.3f} | Số epoch: {res_scaled.n_iter}")

    print(">> Không dùng StandardScaler (Dữ liệu gốc):")
    _, res_raw = run_experiment(threshold=2.0, use_scaler=False)
    print(f"   Accuracy: {res_raw.metrics['accuracy']:.3f} | Số epoch: {res_raw.n_iter}")

if __name__ == "__main__":
    main()