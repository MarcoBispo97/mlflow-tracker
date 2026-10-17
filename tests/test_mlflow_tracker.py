"""Tests for MLflowTracker, each one using an empty MLflow project in a temp folder."""

import os

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402

from mlflow_tracker import MLflowTracker, make_paths_relative  # noqa: E402


@pytest.fixture
def split():
    """A small imbalanced dataset (120 rows, 3 features) split 70/30."""
    X, y = make_classification(
        n_samples=120, n_features=3, n_informative=2, n_redundant=0, weights=[0.6], random_state=0
    )
    X = pd.DataFrame(X, columns=["a", "b", "c"])
    return train_test_split(X, pd.Series(y), test_size=0.3, random_state=0, stratify=y)


@pytest.fixture
def tracker(tmp_path, monkeypatch, split):
    """A tracker inside an empty folder with its own ``mlflow.db`` and ``mlruns/``."""
    monkeypatch.chdir(tmp_path)
    # absolute URI: MLflow caches one store per URI, so "sqlite:///mlflow.db" would be shared
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    yield MLflowTracker("exp", *split)
    mlflow.set_tracking_uri(None)


def test_fingerprint_changes_with_name_params_and_data(tracker, split):
    base = tracker.fingerprint("LR", LogisticRegression(C=1.0))
    assert base == tracker.fingerprint("LR", LogisticRegression(C=1.0))
    assert base != tracker.fingerprint("LR", LogisticRegression(C=0.5))
    assert base != tracker.fingerprint("Other", LogisticRegression(C=1.0))

    X_train, X_test, y_train, y_test = split
    other = MLflowTracker("exp", X_train.iloc[1:], X_test, y_train.iloc[1:], y_test)
    assert base != other.fingerprint("LR", LogisticRegression(C=1.0))


def test_run_saves_metrics_charts_and_model(tracker):
    metrics = tracker.run("LR", LogisticRegression())
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0 <= v <= 1 for v in metrics.values())

    run_id = tracker.top_models().loc[0, "run_id"]
    client = mlflow.MlflowClient()
    files = {a.path for folder in ["charts", "shap"] for a in client.list_artifacts(run_id, folder)}
    assert files == {
        "charts/confusion_matrix.png",
        "charts/roc_curve.png",
        "shap/importance.png",
        "shap/beeswarm.png",
        "shap/importance.csv",
    }
    assert {"shap_a", "shap_b", "shap_c"} <= set(client.get_run(run_id).data.metrics)


def test_same_model_is_not_saved_twice(tracker):
    first = tracker.run("LR", LogisticRegression())
    assert tracker.run("LR", LogisticRegression()) == pytest.approx(first)
    assert len(mlflow.search_runs([tracker.experiment.experiment_id])) == 1

    tracker.run("LR", LogisticRegression(C=0.1))  # different hyperparameter -> new run
    assert len(mlflow.search_runs([tracker.experiment.experiment_id])) == 2


def test_top_models_and_overview(tracker):
    tracker.run_all({"LR": LogisticRegression(), "Tree": DecisionTreeClassifier(max_depth=2)})
    top = tracker.save_overview(n=5, metric="f1")
    assert sorted(top["model"]) == ["LR", "Tree"]
    assert top["f1"].is_monotonic_decreasing
    assert os.path.exists("overview.png")


def test_load_best_model_works_with_missing_values(tracker):
    tracker.run("LR", make_pipeline(SimpleImputer(), LogisticRegression()))
    model, best = tracker.load_best_model()
    assert best["model"] == "LR"
    new = pd.DataFrame({"a": [0.1, None], "b": [None, 1.0], "c": [0.0, 0.0]})
    assert set(model.predict(new)) <= {0, 1}


def test_artifact_paths_are_relative(tracker):
    tracker.run("LR", LogisticRegression())
    make_paths_relative()
    run_id = tracker.top_models().loc[0, "run_id"]
    assert mlflow.get_run(run_id).info.artifact_uri.startswith("mlruns/")
    experiment = mlflow.get_experiment(tracker.experiment.experiment_id)
    assert experiment.artifact_location == f"mlruns/{experiment.experiment_id}"
