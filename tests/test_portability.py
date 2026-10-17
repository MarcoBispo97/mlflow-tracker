"""Tests for MLflowPortability."""

import sqlite3

import pytest

from mlflow_tracker import MLflowPortability


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("file:c:/Users/me/project/mlruns/1/abc", "mlruns/1/abc"),
        ("file:///home/me/project/mlruns/1/abc/artifacts", "mlruns/1/abc/artifacts"),
        ("C:\\Users\\me\\project\\mlruns\\1", "mlruns/1"),
        ("/home/me/project/mlruns", "mlruns"),
        ("./mlruns/2", "mlruns/2"),
        ("mlruns/1/abc", "mlruns/1/abc"),  # already relative
        ("s3://bucket/mlruns/1", "s3://bucket/mlruns/1"),  # remote: unchanged
        ("mlflow-artifacts:/1/abc", "mlflow-artifacts:/1/abc"),
        ("", ""),
        (None, None),
    ],
)
def test_relative(path, expected):
    assert MLflowPortability().relative(path) == expected


def _create_database(path):
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE experiments (experiment_id TEXT, artifact_location TEXT)")
        con.execute("CREATE TABLE runs (run_uuid TEXT, artifact_uri TEXT)")
        con.execute("CREATE TABLE logged_models (model_id TEXT, artifact_location TEXT)")
        con.execute("INSERT INTO experiments VALUES ('1', 'file:C:/old/pc/mlruns/1')")
        con.execute("INSERT INTO runs VALUES ('r1', 'file:C:/old/pc/mlruns/1/r1/artifacts')")
        con.execute("INSERT INTO runs VALUES ('r2', 's3://bucket/mlruns/1/r2')")
        con.execute("INSERT INTO logged_models VALUES ('m1', 'mlruns/1/models/m1')")


def test_fix_makes_every_local_path_relative(tmp_path):
    database = tmp_path / "mlflow.db"
    _create_database(database)
    mlmodel = tmp_path / "mlruns" / "1" / "models" / "m1" / "MLmodel"
    mlmodel.parent.mkdir(parents=True)
    mlmodel.write_text("artifact_path: file:C:/old/pc/mlruns/1/models/m1\n", encoding="utf-8")

    portability = MLflowPortability(database, tmp_path / "mlruns")
    assert portability.fix(verbose=False) == 3  # 2 database rows + 1 MLmodel file
    assert portability.absolute_paths() == []
    assert mlmodel.read_text(encoding="utf-8") == "artifact_path: mlruns/1/models/m1\n"
    with sqlite3.connect(database) as con:
        assert con.execute("SELECT artifact_uri FROM runs WHERE run_uuid = 'r2'").fetchone() == (
            "s3://bucket/mlruns/1/r2",
        )


def test_fix_is_idempotent(tmp_path):
    database = tmp_path / "mlflow.db"
    _create_database(database)
    portability = MLflowPortability(database, tmp_path / "mlruns")
    portability.fix(verbose=False)
    assert portability.fix(verbose=False) == 0


def test_fix_without_database(tmp_path):
    assert MLflowPortability(tmp_path / "missing.db").fix() == 0
