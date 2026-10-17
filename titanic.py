"""Example: train 5 classifiers on the Titanic dataset and track them with MLflow.

Usage:
    python titanic.py
"""

import os
from pathlib import Path

# quieter terminal: no download progress bars or MLflow hints
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from mlflow_tracker import MLflowTracker

FEATURES = ["Pclass", "Sex", "Age"]
TARGET = "Survived"


def prepare(df):
    """Returns the features with ``Sex`` as 0/1 (missing ages are filled inside the pipelines)."""
    X = df[FEATURES].copy()
    X["Sex"] = X["Sex"].map({"male": 0, "female": 1})
    return X.astype(float)


def create_models(positive_weight):
    """Creates 5 pipelines for a SMALL dataset with IMBALANCED classes.

    Every pipeline fills missing values with the median and scales the features.
    The models give more weight to the minority class (survived) and are kept
    simple, so they do not memorize the training set.
    """
    seed = 42
    return {
        "Polynomial Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            PolynomialFeatures(degree=2, include_bias=False),
            LogisticRegression(C=0.5, class_weight="balanced", random_state=seed),
        ),
        "SVM": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SVC(C=1.2, class_weight="balanced", probability=True, random_state=seed),
        ),
        "Decision Tree": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            DecisionTreeClassifier(
                max_depth=4, min_samples_leaf=4, class_weight="balanced", random_state=seed
            ),
        ),
        "LightGBM": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LGBMClassifier(
                n_estimators=150,
                learning_rate=0.05,
                num_leaves=8,
                min_child_samples=10,
                class_weight="balanced",
                random_state=seed,
                verbose=-1,
            ),
        ),
        "XGBoost": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            XGBClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=5,
                scale_pos_weight=positive_weight,
                random_state=seed,
            ),
        ),
    }


def main():
    """Trains and tracks the models, saves the overview and predicts four new passengers."""
    os.chdir(Path(__file__).resolve().parent)  # mlflow.db and mlruns/ live next to this file
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # 1. data: 60% train / 40% test, same share of survivors in both
    df = pd.read_csv("data/titanic.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        prepare(df), df[TARGET], test_size=0.40, random_state=42, stratify=df[TARGET]
    )
    positive_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # 2. train and save every model
    tracker = MLflowTracker(
        "titanic", X_train, X_test, y_train, y_test, labels=("Died", "Survived")
    )
    results = tracker.run_all(create_models(positive_weight))
    print(results.round(3).to_string(index=False), "\n")

    # 3. compare the top 5 by F1
    top = tracker.save_overview(n=5, metric="f1")
    print(top.drop(columns=["run_id", "model_id"]).round(3).to_string(index=False), "\n")

    # 4. use the best model on new passengers
    model, best = tracker.load_best_model(metric="f1")
    print(f"Best model: {best['model']} (F1 = {best['f1']:.3f})")
    new_passengers = pd.DataFrame(
        {
            "Pclass": [1, 3, 2, 3],
            "Sex": ["female", "male", "male", "female"],
            "Age": [29, 22, None, 5],  # missing age: the pipeline fills it with the median
        }
    )
    X_new = prepare(new_passengers)
    new_passengers["survives"] = model.predict(X_new).astype(bool)
    new_passengers["probability"] = model.predict_proba(X_new)[:, 1].round(3)
    print(new_passengers.to_string(index=False))
    print("\nOpen the MLflow UI with:  mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
