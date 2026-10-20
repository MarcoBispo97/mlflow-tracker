"""ClassificationTracker: tracks binary classifiers (target 0/1) in MLflow."""

import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import seaborn as sns
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from utils import BaseTracker, log_chart, log_dataframe, slug


class ClassificationTracker(BaseTracker):
    """Tracks binary classifiers. The model must have ``predict_proba``.

    Args:
        experiment_name: Name of the MLflow experiment.
        X_train: Training features.
        X_test: Test features.
        y_train: Training target (0/1).
        y_test: Test target (0/1).
        labels: Names of the classes 0 and 1, shown in the charts.
        data_source: Where the data came from, e.g. ``"data/titanic.csv"``.
    """

    METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    BEST = "f1"

    def __init__(
        self, experiment_name, X_train, X_test, y_train, y_test, labels=("0", "1"), data_source=None
    ):
        super().__init__(experiment_name, X_train, X_test, y_train, y_test, data_source)
        self.labels = list(labels)

    def compute_metrics(self, model, X, y):
        """Accuracy, precision, recall, F1 and ROC AUC of the model on ``X``."""
        y_pred = model.predict(X)
        y_score = model.predict_proba(X)[:, 1]  # probability of class 1
        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y, y_score),
        }

    def log_charts(self, model):
        """Logs the confusion matrix, the ROC curve and the wrong predictions on the test set."""
        y_pred = model.predict(self.X_test)
        y_score = model.predict_proba(self.X_test)[:, 1]
        self.log_wrong_predictions(y_pred, y_score)

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            confusion_matrix(self.y_test, y_pred),
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.labels,
            yticklabels=self.labels,
            ax=ax,
        )
        ax.set(title="Confusion matrix", xlabel="Predicted", ylabel="Actual")
        log_chart(fig, "charts/confusion_matrix.png")

        fig, ax = plt.subplots(figsize=(5, 5))
        RocCurveDisplay.from_predictions(self.y_test, y_score, plot_chance_level=True, ax=ax)
        ax.set_title("ROC curve")
        log_chart(fig, "charts/roc_curve.png")

    def log_wrong_predictions(self, y_pred, y_score):
        """Saves the test rows that the model got wrong, to study its mistakes in MLflow.

        They appear in three places of the run: a pandas-style table in Artifacts
        (``errors/wrong_predictions.html``, plus a CSV), the ``<project>_wrong_predictions``
        dataset in the Datasets field, and the ``n_wrong`` metric.
        """
        wrong = self.X_test.copy()
        wrong["actual"] = self.y_test
        wrong["predicted"] = y_pred
        wrong["probability"] = y_score.round(3)
        wrong = wrong[wrong["actual"] != wrong["predicted"]].reset_index(names="row")

        mlflow.log_metric("n_wrong", len(wrong))
        if wrong.empty:
            return
        log_dataframe(
            wrong,
            "errors/wrong_predictions",
            f"Wrong predictions on the test set ({len(wrong)} rows)",
        )
        dataset = mlflow.data.from_pandas(
            wrong,
            name=f"{slug(self.experiment.name)}_wrong_predictions",
            targets="actual",
            source=self.data_source,
        )
        mlflow.log_input(dataset, context="wrong_predictions")

    def predict_for_shap(self, model, X):
        """SHAP explains the probability of class 1."""
        return model.predict_proba(X)[:, 1]
