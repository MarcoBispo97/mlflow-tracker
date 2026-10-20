"""Tests for ClassificationTracker and RegressionTracker, each in an empty MLflow project."""

import json
import os
import re

import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MPLBACKEND", "Agg")  # charts only go to files: no window backend (Tk)

import mlflow  # noqa: E402

from classification import ClassificationTracker  # noqa: E402
from regression import RegressionTracker  # noqa: E402
from utils import make_paths_relative  # noqa: E402


@pytest.fixture(autouse=True)
def mlflow_project(tmp_path, monkeypatch):
    """Runs each test inside an empty folder with its own ``mlflow.db`` and ``mlruns/``."""
    monkeypatch.chdir(tmp_path)
    # absolute URI: MLflow caches one store per URI, so "sqlite:///mlflow.db" would be shared
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    yield
    mlflow.set_tracking_uri(None)


def split(X, y):
    """Splits 70/30 and returns pandas objects with columns a, b, c."""
    return train_test_split(
        pd.DataFrame(X, columns=["a", "b", "c"]), pd.Series(y), test_size=0.3, random_state=0
    )


@pytest.fixture
def classifier():
    """A ClassificationTracker on a small imbalanced dataset."""
    X, y = make_classification(
        n_samples=200, n_features=3, n_informative=2, n_redundant=0, weights=[0.6], random_state=0
    )
    return ClassificationTracker("clf", *split(X, y), data_source="data/clf.csv")


@pytest.fixture
def regressor():
    """A RegressionTracker on a small dataset."""
    X, y = make_regression(n_samples=120, n_features=3, noise=10, random_state=0)
    return RegressionTracker("reg", *split(X, y))


def artifacts(run_id):
    """Paths of the charts and SHAP files of a run."""
    client = mlflow.MlflowClient()
    return {a.path for folder in ["charts", "shap"] for a in client.list_artifacts(run_id, folder)}


SHAP_FILES = {"shap/importance.png", "shap/beeswarm.png", "shap/importance.csv"}


def test_classification_run_saves_metrics_and_charts(classifier):
    metrics = classifier.run("LR", LogisticRegression())
    assert set(metrics) == set(ClassificationTracker.METRICS)
    assert all(0 <= v <= 1 for v in metrics.values())
    run_id = classifier.top_models().loc[0, "run_id"]
    assert artifacts(run_id) == {
        "charts/confusion_matrix.png",
        "charts/roc_curve.png",
        "charts/learning_curve.png",
        *SHAP_FILES,
    }


def test_regression_run_saves_metrics_and_charts(regressor):
    metrics = regressor.run("Linear", LinearRegression())
    assert set(metrics) == {"mae", "rmse", "r2"}
    assert metrics["r2"] > 0.9  # the data is almost linear
    run_id = regressor.top_models().loc[0, "run_id"]
    assert artifacts(run_id) == {
        "charts/predicted_vs_actual.png",
        "charts/residuals.png",
        "charts/residuals_vs_predicted.png",
        "charts/distribution.png",
        "charts/learning_curve.png",
        *SHAP_FILES,
    }


def test_run_name_and_datasets(classifier):
    classifier.run("K-Nearest Neighbors", LogisticRegression())
    run = mlflow.get_run(classifier.top_models().loc[0, "run_id"])
    # project + model + date and time, joined by underscores
    assert re.fullmatch(r"clf_k_nearest_neighbors_\d{8}_\d{6}", run.info.run_name)
    assert run.data.tags["model"] == "K-Nearest Neighbors"

    datasets = {d.dataset.name: d.dataset for d in run.inputs.dataset_inputs}
    assert set(datasets) == {"clf_training", "clf_testing"}
    assert json.loads(datasets["clf_training"].source) == {"uri": "data/clf.csv"}  # relative


def test_learning_curve_has_one_point_per_size(regressor):
    regressor.run("Linear", LinearRegression())
    run_id = regressor.top_models().loc[0, "run_id"]
    history = mlflow.MlflowClient().get_metric_history(run_id, "learning_test_rmse")
    n_train = len(regressor.X_train)
    assert sorted(p.step for p in history) == [int(n_train * f) for f in (0.2, 0.4, 0.6, 0.8, 1.0)]


def test_fingerprint_changes_with_name_params_and_data(classifier):
    base = classifier.fingerprint("LR", LogisticRegression(C=1.0))
    assert base == classifier.fingerprint("LR", LogisticRegression(C=1.0))
    assert base != classifier.fingerprint("LR", LogisticRegression(C=0.5))
    assert base != classifier.fingerprint("Other", LogisticRegression(C=1.0))

    c = classifier
    other = ClassificationTracker("clf", c.X_train.iloc[1:], c.X_test, c.y_train.iloc[1:], c.y_test)
    assert base != other.fingerprint("LR", LogisticRegression(C=1.0))


def test_same_model_is_not_saved_twice(classifier):
    first = classifier.run("LR", LogisticRegression())
    assert classifier.run("LR", LogisticRegression()) == pytest.approx(first)
    assert len(mlflow.search_runs([classifier.experiment.experiment_id])) == 1

    classifier.run("LR", LogisticRegression(C=0.1))  # different hyperparameter -> new run
    assert len(mlflow.search_runs([classifier.experiment.experiment_id])) == 2


def test_classification_ranks_higher_f1_first(classifier):
    classifier.run_all({"LR": LogisticRegression(), "Tree": DecisionTreeClassifier(max_depth=1)})
    assert classifier.top_models()["f1"].is_monotonic_decreasing


def test_regression_ranks_lower_rmse_first(regressor):
    regressor.run_all({"Linear": LinearRegression(), "Tree": DecisionTreeRegressor(max_depth=2)})
    top = regressor.top_models()
    assert list(top["model"]) == ["Linear", "Tree"]
    assert top["rmse"].is_monotonic_increasing
    assert regressor.plot_overview().axes[0].get_title() == "Top 2 models by rmse"


def test_run_all_saves_one_comparison_per_top(regressor):
    def comparison_runs():
        runs = mlflow.search_runs([regressor.experiment.experiment_id])
        return runs[runs["tags.top_run_ids"].notna()]

    models = {"Linear": LinearRegression(), "Tree": DecisionTreeRegressor(max_depth=2)}
    regressor.run_all(models)
    assert len(comparison_runs()) == 1
    run_id = comparison_runs().iloc[0]["run_id"]
    files = {a.path for a in mlflow.MlflowClient().list_artifacts(run_id, "comparison")}
    assert files == {
        "comparison/top_models.csv",
        "comparison/top_models.png",
        "comparison/metrics_heatmap.png",
        "comparison/ranking.png",
        "comparison/learning_curves.png",
        "comparison/train_time.png",
        "comparison/shap_importance.png",
    }
    assert len(regressor.top_models()) == 2  # the comparison run is not a model

    regressor.run_all(models)  # nothing changed -> no new comparison
    assert len(comparison_runs()) == 1

    regressor.run_all({"Tree 3": DecisionTreeRegressor(max_depth=3)})  # new model in the top 5
    assert len(comparison_runs()) == 2


def test_load_best_model_works_with_missing_values(regressor):
    regressor.run("Linear", make_pipeline(SimpleImputer(), LinearRegression()))
    model, best = regressor.load_best_model()
    assert best["model"] == "Linear"
    params = mlflow.get_run(best["run_id"]).data.params
    assert params["pipeline"] == "SimpleImputer -> LinearRegression"
    assert params["simpleimputer__strategy"] == "mean"  # every hyperparameter is saved
    assert params["linearregression__fit_intercept"] == "True"
    new = pd.DataFrame({"a": [0.1, None], "b": [None, 1.0], "c": [0.0, 0.0]})
    assert len(model.predict(new)) == 2


def test_artifact_paths_are_relative(classifier):
    classifier.run("LR", LogisticRegression())
    make_paths_relative()
    run_id = classifier.top_models().loc[0, "run_id"]
    assert mlflow.get_run(run_id).info.artifact_uri.startswith("mlruns/")
    experiment = mlflow.get_experiment(classifier.experiment.experiment_id)
    assert experiment.artifact_location == f"mlruns/{experiment.experiment_id}"
