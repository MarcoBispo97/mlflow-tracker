"""Example: train 5 classifiers on the Titanic dataset and track them with MLflow.

Usage:
    python titanic.py           # train, save to MLflow, build the overview, predict
    python titanic.py --show    # same, and also open every chart in the browser
"""

import argparse
import os
from pathlib import Path

# quieter terminal: no download progress bars or MLflow hints
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from mlflow_tracker import DataSplit, MLflowTracker, PlotlyCharts

FEATURES = ["Pclass", "Sex", "Age"]
TARGET = "Survived"
DATA_FILE = "data/titanic.csv"


def prepare(df):
    """Applies the transformation shared by training and new passengers.

    Maps ``Sex`` to 0/1. Missing ages and scaling are handled inside each Pipeline.

    Args:
        df: Raw Titanic rows.

    Returns:
        pandas.DataFrame: A copy with ``Sex`` as 0/1.
    """
    X = df.copy()
    X["Sex"] = X["Sex"].map({"male": 0, "female": 1})
    return X


def create_models(data, random_state=42):
    """Creates 5 pipelines tuned for a SMALL dataset with IMBALANCED classes.

    - Imbalance: more weight for the minority class (survived).
    - Small data: simpler models and regularization (avoids memorizing the training set).

    Args:
        data: The ``DataSplit``; gives the features and the weight of the positive class.
        random_state: Seed of every model.

    Returns:
        dict[str, sklearn.pipeline.Pipeline]: ``{name: pipeline}``.
    """

    def pipeline(*steps):
        preprocessor = ColumnTransformer(
            [
                (
                    "numeric",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    data.features,
                )
            ]
        )
        return Pipeline([("preprocessor", preprocessor), *steps])

    return {
        "Polynomial Logistic Regression": pipeline(
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            (
                "classifier",
                LogisticRegression(
                    C=0.5, class_weight="balanced", max_iter=200, random_state=random_state
                ),
            ),
        ),
        "SVM": pipeline(
            (
                "classifier",
                SVC(
                    C=1.2,
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=random_state,
                ),
            )
        ),
        "Decision Tree": pipeline(
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=4,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            )
        ),
        "LightGBM": pipeline(
            (
                "classifier",
                LGBMClassifier(
                    n_estimators=150,
                    learning_rate=0.05,
                    num_leaves=8,
                    max_depth=7,
                    min_child_samples=10,
                    subsample=0.8,
                    subsample_freq=1,
                    reg_lambda=1.0,
                    class_weight="balanced",
                    random_state=random_state,
                    verbose=-1,
                ),
            )
        ),
        "XGBoost": pipeline(
            (
                "classifier",
                XGBClassifier(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=5,
                    min_child_weight=3,
                    subsample=0.8,
                    reg_lambda=1.0,
                    scale_pos_weight=data.positive_weight,
                    eval_metric="logloss",
                    random_state=random_state,
                ),
            )
        ),
    }


def main():
    """Trains, tracks, builds the overview and predicts four new passengers."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--show", action="store_true", help="open the charts in the browser")
    args = parser.parse_args()

    # mlflow.db and mlruns/ always live next to this file
    os.chdir(Path(__file__).resolve().parent)

    # 1. data
    df = prepare(pd.read_csv(DATA_FILE))
    data = DataSplit.split(df, FEATURES, TARGET, test_size=0.40, random_state=42, stratify=True)
    print(data.summary(), "\n")

    # 2. train and track
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    tracker = MLflowTracker("titanic", data, PlotlyCharts(labels=("Died", "Survived")))
    print(tracker.run_all(create_models(data), show=args.show).to_string(index=False), "\n")

    # 3. overview of the top 5 by F1
    print(
        tracker.save_overview(n=5, metric="f1", show=args.show).round(3).to_string(index=False),
        "\n",
    )

    # 4. inference with the .pkl of the best model
    model, best = tracker.load_best_model(metric="f1")
    print(f"Best model: {best['tags.model_name']} (F1 = {best['metrics.f1']:.3f})")
    new_passengers = pd.DataFrame(
        {
            "Pclass": [1, 3, 2, 3],
            "Sex": ["female", "male", "male", "female"],
            "Age": [29, 22, None, 5],  # missing age: the pipeline fills it with the median
        }
    )
    X_new = prepare(new_passengers)[FEATURES].astype(float)
    result = new_passengers.copy()
    result["survives"] = model.predict(X_new).astype(bool)
    result["probability"] = model.predict_proba(X_new)[:, 1].round(3)
    print(result.to_string(index=False))
    print("\nOpen the MLflow UI with:  mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
