"""Train scikit-learn classifiers and save everything about them to MLflow.

Example:
    tracker = MLflowTracker("my_experiment", X_train, X_test, y_train, y_test)
    tracker.run_all({"Logistic Regression": LogisticRegression()})
    tracker.save_overview(n=5, metric="f1")
    model, best_run = tracker.load_best_model(metric="f1")
"""

import hashlib
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from mlflow.models import infer_signature
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


class MLflowTracker:
    """Trains binary classifiers and saves each one to MLflow, never the same model twice.

    Works with any scikit-learn model or ``Pipeline`` that has ``predict_proba``.
    The target must be 0/1 and the features numeric.

    Args:
        experiment_name: Name of the MLflow experiment (created if it does not exist).
        X_train: Training features.
        X_test: Test features.
        y_train: Training target.
        y_test: Test target.
        labels: Names of the classes 0 and 1, shown in the charts.
    """

    def __init__(self, experiment_name, X_train, X_test, y_train, y_test, labels=("0", "1")):
        self.experiment = mlflow.set_experiment(experiment_name)
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.labels = list(labels)
        make_paths_relative()

    # ---------- training ----------
    def run_all(self, models):
        """Runs every model of a ``{name: model}`` dictionary.

        Returns:
            pandas.DataFrame: One row per model with its test metrics.
        """
        return pd.DataFrame(
            [{"model": name, **self.run(name, model)} for name, model in models.items()]
        )

    def run(self, name, model):
        """Trains, evaluates and saves one model in its own MLflow run.

        If the same model (same name, hyperparameters and data) was already saved,
        nothing is trained and the saved metrics are returned.

        Returns:
            dict[str, float]: The test metrics.
        """
        fingerprint = self.fingerprint(name, model)
        previous = mlflow.search_runs(
            [self.experiment.experiment_id], filter_string=f"tags.fingerprint = '{fingerprint}'"
        )
        if not previous.empty:
            print(f"[{name}] nothing changed -> skipped")
            return {m: previous.iloc[0][f"metrics.{m}"] for m in METRICS}

        with mlflow.start_run(run_name=name):
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            y_score = model.predict_proba(self.X_test)[:, 1]  # probability of class 1

            metrics = {
                "accuracy": accuracy_score(self.y_test, y_pred),
                "precision": precision_score(self.y_test, y_pred),
                "recall": recall_score(self.y_test, y_pred),
                "f1": f1_score(self.y_test, y_pred),
                "roc_auc": roc_auc_score(self.y_test, y_score),
            }
            mlflow.log_metrics(metrics)
            mlflow.log_params(
                {
                    "features": ", ".join(self.X_train.columns),
                    "n_train": len(self.X_train),
                    "n_test": len(self.X_test),
                }
            )

            # float columns accept missing values when the model is used later
            example = self.X_train.head(5).astype(float)
            info = mlflow.sklearn.log_model(
                model,
                name="model",
                input_example=example,
                signature=infer_signature(example, model.predict(example)),
                # the default format (skops) refuses to load tree models such as DecisionTree
                serialization_format="cloudpickle",
            )
            mlflow.set_tags({"fingerprint": fingerprint, "model_id": info.model_id})

            self._log_chart(self.plot_confusion_matrix(y_pred), "charts/confusion_matrix.png")
            self._log_chart(self.plot_roc_curve(y_score, name), "charts/roc_curve.png")
            self.log_shap(model)

        make_paths_relative()
        print(f"[{name}] saved")
        return metrics

    def fingerprint(self, name, model):
        """Short hash of everything that defines a model: name, hyperparameters and data.

        If any of them changes, the fingerprint changes and the model is trained again.
        """
        text = name + str(model.get_params())
        for part in (self.X_train, self.X_test, self.y_train, self.y_test):
            text += part.to_csv()
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    # ---------- charts ----------
    def plot_confusion_matrix(self, y_pred):
        """Heatmap of the confusion matrix (rows = actual, columns = predicted)."""
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
        return fig

    def plot_roc_curve(self, y_score, name):
        """ROC curve of one model, with the random-classifier line."""
        fig, ax = plt.subplots(figsize=(5, 5))
        RocCurveDisplay.from_predictions(
            self.y_test, y_score, name=name, plot_chance_level=True, ax=ax
        )
        ax.set_title("ROC curve")
        return fig

    def log_shap(self, model):
        """Explains the model with SHAP and saves two charts and a CSV.

        Works with any model, because SHAP only uses ``predict_proba``.
        The mean |SHAP| of each feature is also saved as a ``shap_<feature>`` metric.
        """

        def predict(X):  # SHAP passes a numpy array; pipelines need the column names
            return model.predict_proba(pd.DataFrame(X, columns=self.X_train.columns))[:, 1]

        background = self.X_train.sample(min(100, len(self.X_train)), random_state=42)
        sample = self.X_test.sample(min(200, len(self.X_test)), random_state=42)
        explanation = shap.Explainer(predict, background)(sample)

        importance = pd.DataFrame(
            {
                "feature": self.X_train.columns,
                "mean_abs_shap": np.abs(explanation.values).mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False)
        mlflow.log_text(importance.to_csv(index=False), "shap/importance.csv")
        mlflow.log_metrics(
            {f"shap_{row.feature}": row.mean_abs_shap for row in importance.itertuples()}
        )

        fig, ax = plt.subplots(figsize=(6, 3))
        sns.barplot(data=importance, x="mean_abs_shap", y="feature", ax=ax)
        ax.set(title="Feature importance (mean |SHAP|)", xlabel="mean |SHAP|", ylabel="")
        self._log_chart(fig, "shap/importance.png")

        # each dot is one passenger: position = effect on the prediction, color = feature value
        plt.figure()
        shap.plots.beeswarm(explanation, show=False)
        self._log_chart(plt.gcf(), "shap/beeswarm.png")

    @staticmethod
    def _log_chart(fig, path):
        """Saves a matplotlib figure to the current MLflow run and closes it."""
        fig.tight_layout()
        mlflow.log_figure(fig, path)
        plt.close(fig)

    # ---------- best models ----------
    def top_models(self, n=5, metric="f1"):
        """Finds the n best models of the experiment (best run of each model name).

        Returns:
            pandas.DataFrame: ``model``, the metrics, ``run_id`` and ``model_id``, best first.
        """
        runs = mlflow.search_runs(
            [self.experiment.experiment_id], order_by=[f"metrics.{metric} DESC"]
        )
        if runs.empty:
            return pd.DataFrame(columns=["model", *METRICS, "run_id", "model_id"])
        table = pd.DataFrame({"model": runs["tags.mlflow.runName"]})
        for m in METRICS:
            table[m] = runs[f"metrics.{m}"]
        table["run_id"] = runs["run_id"]
        table["model_id"] = runs["tags.model_id"]
        return table.drop_duplicates("model").head(n).reset_index(drop=True)

    def load_best_model(self, metric="f1"):
        """Loads the best model of the experiment, ready for ``predict``.

        Returns:
            tuple: The fitted model and its row of ``top_models()``.
        """
        best = self.top_models(n=1, metric=metric).iloc[0]
        return mlflow.sklearn.load_model(f"models:/{best['model_id']}"), best

    def save_overview(self, n=5, metric="f1", file="overview.png"):
        """Saves a bar chart that compares the metrics of the top n models.

        Returns:
            pandas.DataFrame: The ranking returned by ``top_models()``.
        """
        top = self.top_models(n, metric)
        scores = top.melt(
            id_vars="model", value_vars=METRICS, var_name="metric", value_name="score"
        )
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=scores, x="metric", y="score", hue="model", ax=ax)
        ax.set(title=f"Top {len(top)} models by {metric}", ylim=(0, 1), xlabel="")
        ax.legend(loc="lower right", fontsize="small")
        fig.tight_layout()
        fig.savefig(file, dpi=120)
        plt.close(fig)
        print(f"Overview chart saved to: {file}")
        return top


def make_paths_relative(database="mlflow.db"):
    """Makes the artifact paths saved in the MLflow database relative.

    MLflow saves absolute paths (e.g. ``file:C:/Users/me/project/mlruns/1/...``), which
    break when the project is moved to another folder or computer. This turns them into
    ``mlruns/1/...``, so MLflow (the script or ``mlflow ui``) must be started from the
    project folder.
    """
    if not Path(database).exists():
        return
    project = Path.cwd()
    prefixes = [f"file:{project.as_posix()}/", f"{project.as_uri()}/"]  # Windows and Linux
    columns = [
        ("experiments", "artifact_location"),
        ("runs", "artifact_uri"),
        ("logged_models", "artifact_location"),
    ]
    with sqlite3.connect(database) as con:
        for table, column in columns:
            for prefix in prefixes:
                con.execute(f"UPDATE {table} SET {column} = REPLACE({column}, ?, '')", (prefix,))
