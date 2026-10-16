import hashlib
import json
import os
import re
import tempfile
import unicodedata
from datetime import datetime

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from .charts import PlotlyCharts
from .overview import Overview
from .portability import MLflowPortability


class MLflowTracker:
    """Trains binary classifiers and saves everything to MLflow: the model, a .pkl,
    metrics, charts, SHAP, features and the run history, without saving the same model twice.
    Works with any scikit-learn compatible model or Pipeline."""

    METRICS = ["accuracy", "precision", "recall", "f1"]

    def __init__(self, experiment_name, data, charts=None):
        experiment = mlflow.set_experiment(experiment_name)
        # MLflow stores an absolute path when it creates the experiment: make it relative
        MLflowPortability().fix()
        self.experiment = mlflow.get_experiment(experiment.experiment_id)
        self.data = data
        self.charts = charts or PlotlyCharts()
        self.history = []

    # ---------- fingerprint (avoids saving the same model twice) ----------
    @staticmethod
    def _model_params(model):
        """Hyperparameters of the model/pipeline (inner estimators become their class name)."""
        def serialize(o):
            if hasattr(o, "get_params"):
                return type(o).__name__
            return getattr(o, "__name__", repr(o))
        return json.dumps(model.get_params(deep=True), default=serialize, sort_keys=True)

    @staticmethod
    def _hash_data(*objs):
        """Hash of the data content (values, columns and index)."""
        h = hashlib.sha256()
        for o in objs:
            h.update(pd.util.hash_pandas_object(o, index=True).values.tobytes())
            h.update(str(getattr(o, "columns", getattr(o, "name", ""))).encode())
        return h.hexdigest()

    def fingerprint(self, name, model):
        """Everything that defines the result: name, class, parameters, features,
        train/test data and split configuration."""
        d = self.data
        content = {
            "model": name,
            "class": type(model).__name__,
            "params": self._model_params(model),
            "features": d.features,
            "data": self._hash_data(d.X_train, d.y_train, d.X_test, d.y_test),
            "split": d.config,
            "metrics": self.METRICS + ["roc_auc"],
        }
        text = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def find_by_fingerprint(self, fingerprint):
        """Returns the training run that already has this fingerprint (or None)."""
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            filter_string=f"tags.kind = 'training' and tags.fingerprint = '{fingerprint}'",
            max_results=1,
        )
        return None if runs.empty else runs.iloc[0]

    # ---------- evaluation method ----------
    def log_methodology(self):
        """Records how the model was evaluated (holdout split)."""
        d = self.data
        total = len(d.X_train) + len(d.X_test)
        mlflow.log_params({
            "evaluation": "holdout (train_test_split)",
            **{f"split_{k}": v for k, v in d.config.items()},
            "n_train": len(d.X_train),
            "n_test": len(d.X_test),
            "test_share": round(len(d.X_test) / total, 4),
            "positive_rate_train": round(float(d.y_train.mean()), 4),
            "positive_rate_test": round(float(d.y_test.mean()), 4),
        })

    # ---------- model ----------
    @staticmethod
    def _slug(text):
        """'Decision Tree' -> 'decision_tree' (lowercase, no accents, no spaces)."""
        ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
        return re.sub(r"[^a-z0-9]+", "_", ascii_text.lower()).strip("_")

    def model_name(self, name, moment):
        """Traceable name: experiment + model + date/time.
        E.g. titanic_svm_20261016_101530"""
        return f"{self._slug(self.experiment.name)}_{self._slug(name)}_{moment:%Y%m%d_%H%M%S}"

    def log_model(self, model, model_name, X_example):
        """Saves the model to MLflow (with signature) and a .pkl with the date/time in its name."""
        X_example = X_example.astype(float)  # float accepts missing values at inference time
        signature = infer_signature(X_example, model.predict(X_example.head(5)))
        info = mlflow.sklearn.log_model(
            sk_model=model,
            name=model_name,
            input_example=X_example.head(5),
            signature=signature,
            serialization_format="cloudpickle",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, f"{model_name}.pkl")
            joblib.dump(model, path)
            mlflow.log_artifact(path, artifact_path="pkl")
        mlflow.set_tag("model_id", info.model_id)
        return info

    # ---------- features ----------
    def log_features(self, X):
        """Records the names, count and types of the training features."""
        features = list(X.columns)
        mlflow.log_param("n_features", len(features))
        mlflow.log_dict({"features": features, "dtypes": X.dtypes.astype(str).to_dict()},
                        "features/features.json")
        return features

    # ---------- metrics ----------
    def log_metrics(self, y_true, y_pred, y_score=None, model_id=None):
        """Saves the metrics to the run and to the model (shown in the models list)."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
        }
        if y_score is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        mlflow.log_metrics(metrics, model_id=model_id)
        return metrics

    # ---------- charts (Plotly -> interactive in MLflow) ----------
    def log_confusion_matrix(self, y_true, y_pred, name, show=False):
        fig = self.charts.confusion_matrix(y_true, y_pred, title=f"Confusion matrix - {name}")
        mlflow.log_figure(fig, "charts/confusion_matrix.html")
        if show:
            fig.show()

    def log_roc_curve(self, y_true, y_score, name, show=False):
        fig = self.charts.roc_curve(y_true, y_score, name)
        mlflow.log_figure(fig, "charts/roc_curve.html")
        if show:
            fig.show()

    # ---------- feature importance (SHAP) ----------
    def log_shap(self, model, name, n_samples=200, show=False):
        """Computes SHAP for each feature and saves shap/importance.html, shap/scatter.html
        and shap/importance.csv. Works with any model/pipeline, because SHAP only uses predictions."""
        X_train, X_test = self.data.X_train, self.data.X_test

        def predict(X):  # SHAP passes numpy; the pipeline needs the column names
            return self._score(model, pd.DataFrame(X, columns=X_train.columns))

        background = X_train.sample(min(100, len(X_train)), random_state=42)
        sample = X_test.sample(min(n_samples, len(X_test)), random_state=42)
        explainer = shap.Explainer(predict, shap.maskers.Independent(background))
        values = explainer(sample).values

        importance = pd.DataFrame({
            "feature": X_test.columns,
            "mean_abs_shap": np.abs(values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)

        fig_importance = self.charts.shap_importance(importance, name)
        fig_scatter = self.charts.shap_scatter(values, sample, name)
        mlflow.log_figure(fig_importance, "shap/importance.html")
        mlflow.log_figure(fig_scatter, "shap/scatter.html")
        mlflow.log_text(importance.to_csv(index=False), "shap/importance.csv")
        # also as metrics, to compare feature importance between models in the UI
        mlflow.log_metrics({f"shap_{f}": v for f, v in zip(importance["feature"], importance["mean_abs_shap"])})
        if show:
            fig_importance.show()
            fig_scatter.show()
        return importance

    @staticmethod
    def _score(model, X):
        """Probability of the positive class; falls back to decision_function."""
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        if hasattr(model, "decision_function"):
            return model.decision_function(X)
        return None

    # ---------- run ----------
    def log_run_info(self, name, model_name, start, end, fingerprint):
        """Saves the model name, the traceable name and the start/end time."""
        record = {
            "model": name,
            "model_name": model_name,
            "status": "new",
            "start": f"{start:%Y-%m-%d %H:%M:%S}",
            "end": f"{end:%Y-%m-%d %H:%M:%S}",
            "duration_s": round((end - start).total_seconds(), 3),
        }
        mlflow.set_tags({
            "kind": "training", "model": name, "model_name": model_name, "fingerprint": fingerprint,
            "start": record["start"], "end": record["end"],
        })
        mlflow.log_metric("duration_s", record["duration_s"])
        self.history.append(record)
        return record

    def run(self, name, model, show=False):
        """Trains, evaluates and saves everything in one MLflow run (run name = model name).
        If a run with the same model/parameters/features/data exists, nothing is saved again."""
        d = self.data
        fingerprint = self.fingerprint(name, model)
        existing = self.find_by_fingerprint(fingerprint)
        if existing is not None:
            print(f"[{name}] nothing changed -> already saved as {existing['tags.model_name']} (no new run)")
            self.history.append({
                "model": name, "model_name": existing["tags.model_name"], "status": "unchanged (skipped)",
                "start": existing["tags.start"], "end": existing["tags.end"],
                "duration_s": existing.get("metrics.duration_s"),
            })
            return {m: existing[f"metrics.{m}"] for m in self.METRICS + ["roc_auc"]
                    if f"metrics.{m}" in existing}

        with mlflow.start_run(run_name=name):
            start = datetime.now()
            model.fit(d.X_train, d.y_train)
            predictions = model.predict(d.X_test)
            end = datetime.now()

            model_name = self.model_name(name, start)
            y_score = self._score(model, d.X_test)

            self.log_methodology()
            self.log_features(d.X_train)
            info = self.log_model(model, model_name, d.X_train)
            metrics = self.log_metrics(d.y_test, predictions, y_score, info.model_id)
            self.log_confusion_matrix(d.y_test, predictions, name, show)
            if y_score is not None:
                self.log_roc_curve(d.y_test, y_score, name, show)
                self.log_shap(model, name, show=show)
            self.log_run_info(name, model_name, start, end, fingerprint)
            print(f"[{name}] saved as {model_name}")
        return metrics

    def run_all(self, models, show=False):
        """Runs every model of the {name: model} dictionary and returns the summary."""
        for name, model in models.items():
            self.run(name, model, show=show)
        return self.summary()

    def summary(self):
        """The run history as a DataFrame."""
        return pd.DataFrame(self.history)

    # ---------- best models ----------
    def top_models(self, n=5, metric="f1"):
        """The n best training runs of the experiment (all executions) by the metric."""
        return mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            filter_string="tags.kind = 'training'",
            order_by=[f"metrics.{metric} DESC"],
            max_results=n,
        )

    def load_best_model(self, metric="f1"):
        """Downloads and loads the .pkl of the best model. Returns (model, run row).
        Only load .pkl files you created yourself: unpickling runs code."""
        best = self.top_models(n=1, metric=metric).iloc[0]
        path = mlflow.artifacts.download_artifacts(
            run_id=best["run_id"], artifact_path=f"pkl/{best['tags.model_name']}.pkl"
        )
        return joblib.load(path), best

    def save_overview(self, n=5, metric="f1", show=False):
        """Overview page of the top N (see the Overview class)."""
        overview = Overview(self.experiment.experiment_id, self.data, n=n, metric=metric)
        return overview.generate(show=show)
