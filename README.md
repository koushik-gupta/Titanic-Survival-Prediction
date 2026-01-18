
# Titanic Survival Prediction 🚢
![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)


A machine learning classification project that predicts whether a passenger survived the Titanic disaster based on demographic and travel-related features.

The project demonstrates a complete ML pipeline including data preprocessing, feature engineering, model training, evaluation, and a command-line interface (CLI) for real-time user predictions. Multiple models are compared and combined using ensemble (majority voting) logic.



## Features

- Light/dark mode toggle
- Live previews
- Fullscreen mode
- Cross platform

- End-to-end machine learning pipeline
- Data cleaning and feature engineering
- Comparison of multiple classification models
- Model evaluation using accuracy, precision, recall, F1-score
- Confusion matrix and performance visualizations
- CLI-based user input prediction system
- Ensemble prediction using majority voting

## Tech Stack
- **Language:** Python 3
- **Libraries:** NumPy, Pandas, Matplotlib
- **Machine Learning:** scikit-learn
- **Model Persistence:** joblib
- **Version Control:** Git & GitHub


## Dataset
- **Source:** Kaggle Titanic Dataset
- **Total Records:** 891
- **Target Variable:** Survived (0 = Did Not Survive, 1 = Survived)
- **Type:** Binary Classification

## Preprocessing & Feature Engineering
The following preprocessing steps were applied:

- Missing values handled:
  - Age filled using median
  - Embarked filled using mode
- Feature engineering:
  - HasCabin: Binary feature extracted from Cabin column
  - TicketGroupSize: Number of passengers sharing the same ticket
- Encoding:
  - Sex encoded as binary (male = 0, female = 1)
  - Embarked one-hot encoded
- Feature scaling:
  - StandardScaler applied for Logistic Regression and KNN

## Models Used
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Random Forest Classifier

## Evaluation and Results
Models were evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Logistic Regression achieved the best overall performance with the highest F1-score and a balanced trade-off between precision and recall.

Evaluation plots and confusion matrices are available in the `images/` directory.

## User Prediction(CLI)
The project includes a command-line interface where users can enter passenger details such as:

- Passenger class
- Age
- Sex
- Fare
- Family members aboard
- Embarkation port

Each trained model predicts survival independently, and a final prediction is made using majority voting across all models.

## 📁 Project Structure

```text
titanic-survival-prediction/
│
├── data/
│   └── raw/
│       └── train.csv
│
├── src/
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── predict_user.py
│
├── models/
│   └── saved model files (.pkl)
│
├── images/
│   └── evaluation plots
│
├── requirements.txt
└── README.md
```

## Author

**Koushik Gupta**  
B.Tech (Information Technology)
