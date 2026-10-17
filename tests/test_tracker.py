"""Tests for MLflowTracker and Overview, using a real MLflow project in a temp folder."""

from datetime import datetime

import mlflow
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from mlflow_tracker import DataSplit, MLflowPortability, MLflowTracker


def test_slug():
    assert MLflowTracker._slug("Árvore de Decisão") == "arvore_de_decisao"
    assert MLflowTracker._slug("  XGBoost (v2)! ") == "xgboost_v2"


def test_model_name(mlflow_project, data):
    tracker = MLflowTracker("My Experiment", data)
    moment = datetime(2026, 10, 16, 9, 5, 3)
    assert tracker.model_name("Decision Tree", moment) == (
        "my_experiment_decision_tree_20261016_090503"
    )


def test_fingerprint_changes_with_params_and_data(mlflow_project, df, data):
    tracker = MLflowTracker("exp", data)
    base = tracker.fingerprint("LR", LogisticRegression(C=1.0))
    assert base == tracker.fingerprint("LR", LogisticRegression(C=1.0))
    assert base != tracker.fingerprint("LR", LogisticRegression(C=0.5))
    assert base != tracker.fingerprint("Other name", LogisticRegression(C=1.0))

    other_data = DataSplit.split(df, ["a", "b", "c"], "label", test_size=0.3, random_state=1)
    assert base != MLflowTracker("exp", other_data).fingerprint("LR", LogisticRegression(C=1.0))


def test_run_logs_everything(mlflow_project, data):
    tracker = MLflowTracker("exp", data)
    metrics = tracker.run("LR", LogisticRegression())

    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0 <= v <= 1 for v in metrics.values())

    run = tracker.top_models(n=1).iloc[0]
    assert run["tags.kind"] == "training"
    assert run["params.n_train"] == str(len(data.X_train))
    artifacts = {
        a.path
        for folder in ["charts", "shap", "pkl", "features"]
        for a in mlflow.MlflowClient().list_artifacts(run["run_id"], folder)
    }
    assert artifacts >= {
        "charts/confusion_matrix.html",
        "charts/roc_curve.html",
        "shap/importance.html",
        "shap/scatter.html",
        "shap/importance.csv",
        f"pkl/{run['tags.model_name']}.pkl",
        "features/features.json",
    }
    assert {"metrics.shap_a", "metrics.shap_b", "metrics.shap_c"} <= set(run.index)


def test_same_model_is_not_saved_twice(mlflow_project, data):
    tracker = MLflowTracker("exp", data)
    first = tracker.run("LR", LogisticRegression())
    second = tracker.run("LR", LogisticRegression())

    assert second == pytest.approx(first)
    assert list(tracker.summary()["status"]) == ["new", "unchanged (skipped)"]
    assert len(tracker.top_models(n=10)) == 1

    tracker.run("LR", LogisticRegression(C=0.1))  # different hyperparameter -> new run
    assert len(tracker.top_models(n=10)) == 2


def test_load_best_model_predicts(mlflow_project, data):
    tracker = MLflowTracker("exp", data)
    tracker.run_all({"LR": LogisticRegression(), "Tree": DecisionTreeClassifier(max_depth=2)})

    model, best = tracker.load_best_model(metric="f1")
    assert best["metrics.f1"] == tracker.top_models(n=2)["metrics.f1"].max()
    predictions = model.predict(data.X_test)
    assert set(predictions) <= {0, 1}


def test_overview(mlflow_project, data):
    tracker = MLflowTracker("exp", data)
    tracker.run_all({"LR": LogisticRegression(), "Tree": DecisionTreeClassifier(max_depth=2)})

    ranking = tracker.save_overview(n=5, metric="f1")
    assert len(ranking) == 2
    assert ranking["f1"].is_monotonic_decreasing
    assert (mlflow_project / "overview.html").read_text(encoding="utf-8").startswith("<!doctype")

    def overview_runs():
        runs = mlflow.search_runs(experiment_ids=[tracker.experiment.experiment_id])
        return runs[runs["tags.kind"] == "overview"]

    assert len(overview_runs()) == 1
    tracker.save_overview(n=5, metric="f1")  # same top 5 -> no new overview run
    assert len(overview_runs()) == 1


def test_artifact_paths_are_relative(mlflow_project, data):
    MLflowTracker("exp", data).run("LR", LogisticRegression())
    MLflowPortability().fix(verbose=False)  # the tracker fixes paths when it starts
    assert MLflowPortability().absolute_paths() == []


def test_predict_works_with_missing_values(mlflow_project, data):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    tracker = MLflowTracker("exp", data)
    tracker.run("LR", make_pipeline(SimpleImputer(), LogisticRegression()))
    model, _ = tracker.load_best_model()
    new = pd.DataFrame({"a": [0.1, None], "b": [None, 1.0], "c": [0.0, 0.0]})
    assert len(model.predict(new)) == 2
