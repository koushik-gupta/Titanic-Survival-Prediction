import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from preprocessing import preprocess_dataframe

# Load data
df = pd.read_csv("data/raw/train.csv")

# Preprocess
df = preprocess_dataframe(df)

X = df.drop("Survived", axis=1)
y = df["Survived"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale (for LR & KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
model_LR = LogisticRegression()
model_KNN = KNeighborsClassifier(29)
model_DT = DecisionTreeClassifier()
model_RF = RandomForestClassifier()

# Train
model_LR.fit(X_train_scaled, y_train)
model_KNN.fit(X_train_scaled, y_train)
model_DT.fit(X_train, y_train)
model_RF.fit(X_train, y_train)

# Save models
joblib.dump(model_LR, "models/logistic_regression.pkl")
joblib.dump(model_KNN, "models/knn.pkl")
joblib.dump(model_DT, "models/decision_tree.pkl")
joblib.dump(model_RF, "models/random_forest.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Models trained and saved.")
