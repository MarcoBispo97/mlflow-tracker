"""Shared fixtures: a small dataset and an isolated MLflow project per test."""

import os

import pandas as pd
import pytest
from sklearn.datasets import make_classification

os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402

from mlflow_tracker import DataSplit  # noqa: E402


@pytest.fixture
def df():
    """A small imbalanced binary dataset (120 rows, 3 numeric features)."""
    X, y = make_classification(
        n_samples=120, n_features=3, n_informative=2, n_redundant=0, weights=[0.6], random_state=0
    )
    data = pd.DataFrame(X, columns=["a", "b", "c"])
    data["label"] = y
    return data


@pytest.fixture
def data(df):
    """A stratified 70/30 split of ``df``."""
    return DataSplit.split(df, ["a", "b", "c"], "label", test_size=0.3, random_state=0)


@pytest.fixture
def mlflow_project(tmp_path, monkeypatch):
    """Runs the test inside an empty folder with its own ``mlflow.db`` and ``mlruns/``."""
    monkeypatch.chdir(tmp_path)
    # absolute URI: MLflow caches one store per URI, so "sqlite:///mlflow.db" would be shared
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    yield tmp_path
    mlflow.set_tracking_uri(None)
