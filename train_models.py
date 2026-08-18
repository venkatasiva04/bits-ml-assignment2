"""
train_models.py
Run this file in VS Code (Run button or: python train_models.py).

It trains the 5 required classification models on the Wisconsin Breast Cancer
dataset, calculates all 6 evaluation metrics, prints a comparison table, saves
each trained model into the model/ folder, and writes test_data.csv.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42

# Folders (created next to this file)
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
for directory in (MODEL_DIR, REPORTS_DIR, FIGURES_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def load_data():
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target
    print(f"Dataset loaded: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"Meets minimum 500 rows: {X.shape[0] >= 500}")
    print(f"Meets minimum 12 features: {X.shape[1] >= 12}")
    return X, y


def build_pipeline(model):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def evaluate(name, pipeline, X_train, y_train, X_test, y_test):
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    return {
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_test, probabilities),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1": f1_score(y_test, predictions),
        "MCC": matthews_corrcoef(y_test, predictions),
    }, predictions


def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "Logistic Regression": (
            "logistic_regression",
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        ),
        "Decision Tree": ("decision_tree", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        "kNN": ("knn", KNeighborsClassifier(n_neighbors=5)),
        "Naive Bayes": ("naive_bayes", GaussianNB()),
        "Random Forest (Ensemble)": (
            "random_forest",
            RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        ),
    }

    rows = []
    predictions_by_model = {}
    for display_name, (file_stem, estimator) in models.items():
        pipeline = build_pipeline(estimator)
        metrics, predictions = evaluate(
            display_name, pipeline, X_train, y_train, X_test, y_test
        )
        rows.append(metrics)
        predictions_by_model[display_name] = predictions
        joblib.dump(pipeline, MODEL_DIR / f"{file_stem}.joblib")
        print(f"Saved model: model/{file_stem}.joblib")

    comparison = (
        pd.DataFrame(rows).sort_values(by="F1", ascending=False).reset_index(drop=True)
    )
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    print("\n===== Model Comparison Table =====")
    print(comparison.to_string(index=False))

    winner = comparison.iloc[0]["ML Model Name"]
    print(f"\nOverall winner (highest F1): {winner}")

    # Save test data for the Streamlit app (test data only, as required)
    test_data = X_test.copy()
    test_data["target"] = y_test.values
    test_data.to_csv(BASE_DIR / "test_data.csv", index=False)
    print("Saved test_data.csv")

    # Confusion matrix figure for the winning model
    cm = confusion_matrix(y_test, predictions_by_model[winner])
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix - {winner}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "winner_confusion_matrix.png", dpi=150)
    print("Saved reports/figures/winner_confusion_matrix.png")

    print("\nClassification report for the winner:")
    print(classification_report(y_test, predictions_by_model[winner]))


if __name__ == "__main__":
    main()
