import numpy as np
import pandas as pd

def preprocess_dataframe(df):
    df = df.copy()

    df["HasCabin"] = df["Cabin"].notna().astype(int)
    df.drop(columns=["Cabin"], inplace=True)

    df["TicketGroupSize"] = df.groupby("Ticket")["Ticket"].transform("count")

    df.drop(columns=["PassengerId", "Name", "Ticket"], inplace=True)

    df["Age"] = df["Age"].fillna(df["Age"].median())

    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

    return df


def preprocess_user_input(
    Pclass,
    Sex,
    Age,
    SibSp,
    Parch,
    Fare,
    HasCabin,
    TicketGroupSize,
    Embarked
):
    Sex = 1 if Sex.lower() == "female" else 0
    HasCabin = 1 if HasCabin.lower() == "yes" else 0

    Embarked_Q = 1 if Embarked.upper() == "Q" else 0
    Embarked_S = 1 if Embarked.upper() == "S" else 0

    features = np.array([[
        Pclass,
        Sex,
        Age,
        SibSp,
        Parch,
        Fare,
        HasCabin,
        TicketGroupSize,
        Embarked_Q,
        Embarked_S
    ]])

    return features
