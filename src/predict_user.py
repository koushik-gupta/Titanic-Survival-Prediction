import joblib
from preprocessing import preprocess_user_input

# Load models
scaler = joblib.load("models/scaler.pkl")
lr = joblib.load("models/logistic_regression.pkl")
knn = joblib.load("models/knn.pkl")
dt = joblib.load("models/decision_tree.pkl")
rf = joblib.load("models/random_forest.pkl")

# User input
Pclass = int(input("Passenger Class (1/2/3): "))
Sex = input("Sex (male/female): ")
Age = float(input("Age: "))
SibSp = int(input("Siblings/Spouses: "))
Parch = int(input("Parents/Children: "))
Fare = float(input("Fare: "))
HasCabin = input("Cabin known? (yes/no): ")
TicketGroupSize = int(input("Ticket Group Size: "))
Embarked = input("Embarked (C/Q/S): ")

user_features = preprocess_user_input(
    Pclass, Sex, Age, SibSp, Parch,
    Fare, HasCabin, TicketGroupSize, Embarked
)

user_scaled = scaler.transform(user_features)

preds = {
    "Logistic Regression": lr.predict(user_scaled)[0],
    "KNN": knn.predict(user_scaled)[0],
    "Decision Tree": dt.predict(user_features)[0],
    "Random Forest": rf.predict(user_features)[0]
}

for model, pred in preds.items():
    print(model, "→", "Survived" if pred == 1 else "Did NOT Survive")

final = 1 if sum(preds.values()) >= 2 else 0
print("\nFINAL PREDICTION:", "Survived" if final == 1 else "Did NOT Survive")
