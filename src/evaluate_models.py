import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from preprocessing import preprocess_dataframe

# Load data
df = pd.read_csv("data/raw/train.csv")
df = preprocess_dataframe(df)

X = df.drop("Survived", axis=1)
y = df["Survived"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load models
scaler = joblib.load("models/scaler.pkl")
lr = joblib.load("models/logistic_regression.pkl")
knn = joblib.load("models/knn.pkl")
dt = joblib.load("models/decision_tree.pkl")
rf = joblib.load("models/random_forest.pkl")

X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": (lr, X_test_scaled),
    "KNN": (knn, X_test_scaled),
    "Decision Tree": (dt, X_test),
    "Random Forest": (rf, X_test)
}

metrics = {}

for name, (model, X_eval) in models.items():
    y_pred = model.predict(X_eval)

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f"Confusion Matrix - {name}")
    plt.savefig(f"images/cm_{name.replace(' ', '_').lower()}.png")
    plt.close()

    metrics[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }

df_metrics = pd.DataFrame(metrics).T
df_metrics.plot(kind="bar", figsize=(10,6))
plt.title("Model Performance Comparison")
plt.ylim(0,1)
plt.tight_layout()
plt.savefig("images/model_performance_comparison.png")
plt.close()

print(df_metrics)
