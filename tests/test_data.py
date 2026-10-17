"""Tests for DataSplit."""

import pytest

from mlflow_tracker import DataSplit


def test_split_sizes_and_config(df):
    data = DataSplit.split(df, ["a", "b", "c"], "label", test_size=0.25, random_state=1)
    assert len(data.X_train) == 90
    assert len(data.X_test) == 30
    assert data.features == ["a", "b", "c"]
    assert data.config == {"test_size": 0.25, "random_state": 1, "stratify": True}


def test_split_is_stratified(data):
    assert data.y_train.mean() == pytest.approx(data.y_test.mean(), abs=0.02)


def test_split_is_reproducible(df):
    first = DataSplit.split(df, ["a", "b", "c"], "label", random_state=7)
    second = DataSplit.split(df, ["a", "b", "c"], "label", random_state=7)
    assert first.X_train.index.equals(second.X_train.index)


def test_positive_weight(data):
    negatives = (data.y_train == 0).sum()
    positives = (data.y_train == 1).sum()
    assert data.positive_weight == pytest.approx(negatives / positives)


def test_summary(data):
    summary = data.summary()
    assert list(summary.index) == ["train", "test"]
    assert summary.loc["train", "rows"] == len(data.X_train)
