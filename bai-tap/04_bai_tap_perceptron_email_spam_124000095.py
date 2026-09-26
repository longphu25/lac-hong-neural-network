import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)

# 1. Đọc dữ liệu
df = pd.read_csv("data/email_classification.csv")

# 2. Phân chia tập train - test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 3. Tách feature và nhãn
input_fields = [
    "has_url", "email_length", "word_count",
    "char_count", "digit_count", "uppercase_words",
    "exclamations", "avg_word_length", "punc_ratio",
    "has_noreply", "has_free", "has_win",
    "has_winner", "has_click", "has_offer",
    "has_urgent", "has_limited", "has_buy",
    "has_now", "has_money"
]
output_fields = ["label"]

train_x = np.asarray(train_df[input_fields], dtype=float)
train_y = np.asarray(train_df[output_fields], dtype=float)

test_x = np.asarray(test_df[input_fields], dtype=float)
test_y = np.asarray(test_df[output_fields], dtype=float)


# 4. Định nghĩa lớp Perceptron
class Perceptron:
    def __init__(
        self,
        input_size: int,
        weights: np.ndarray | None = None,
        bias: float | None = None,
    ):
        if weights is None:
            self.weights = np.random.uniform(-0.05, 0.05, size=(input_size, 1))
        else:
            assert weights.shape == (input_size, 1), (
                f"input size is {input_size} but weights matrix has shape {weights.shape}"
            )
            self.weights = weights.astype(float)
        self.bias = 0.0 if bias is None else float(bias)

    def forward(self, x: np.ndarray) -> np.ndarray:
        z = x @ self.weights + self.bias
        return (z > 0).astype(int)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 0.1,
        epochs: int = 100,
    ):
        assert y.shape == (x.shape[0], 1), (
            f"y has shape {y.shape}, expected {(x.shape[0], 1)}"
        )
        for _ in range(epochs):
            y_hat = self.forward(x)
            error = y - y_hat
            if np.all(error == 0):
                break
            self.weights += learning_rate * (x.T @ error)
            self.bias += learning_rate * np.sum(error)


# 5. Huấn luyện mô hình gốc
model = Perceptron(input_size=len(input_fields))
model.fit(train_x, train_y, epochs=200)

# 6. Đánh giá mô hình gốc
y_pred = model.forward(test_x).ravel()
y_true = test_y.ravel().astype(int)

print("=== BÁO CÁO PHÂN LOẠI (MÔ HÌNH GỐC) ===")
print(classification_report(y_true, y_pred, target_names=["Not Spam (0)", "Spam (1)"], zero_division=0))

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Spam (0)", "Spam (1)"])
disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
plt.title("Confusion Matrix - Perceptron Model (Test Set)")
plt.tight_layout()
plt.show()

# 7. Cải thiện với Chuẩn hóa đặc trưng (StandardScaler)
scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x)
test_x_scaled = scaler.transform(test_x)

model_scaled = Perceptron(input_size=len(input_fields))
model_scaled.fit(train_x_scaled, train_y, learning_rate=0.001, epochs=100)

y_pred_scaled = model_scaled.forward(test_x_scaled).ravel()

print("\n=== BÁO CÁO PHÂN LOẠI SAU KHI CHUẨN HÓA (SCALED) ===")
print(classification_report(y_true, y_pred_scaled, target_names=["Not Spam (0)", "Spam (1)"], zero_division=0))

cm_scaled = confusion_matrix(y_true, y_pred_scaled, labels=[0, 1])
fig, ax = plt.subplots(figsize=(6, 5))
disp_scaled = ConfusionMatrixDisplay(confusion_matrix=cm_scaled, display_labels=["Not Spam (0)", "Spam (1)"])
disp_scaled.plot(ax=ax, cmap="Greens", colorbar=False, values_format="d")
plt.title("Confusion Matrix - Perceptron sau Chuẩn hóa (Test Set)")
plt.tight_layout()
plt.show()
