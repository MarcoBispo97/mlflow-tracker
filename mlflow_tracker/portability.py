import re
import sqlite3
from pathlib import Path


class MLflowPortability:
    """Makes an MLflow project portable between computers.

    MLflow stores ABSOLUTE artifact paths in the database
    (e.g. file:c:/Users/me/project/mlruns/1/...). On another computer that folder
    does not exist and the UI cannot find models, charts or .pkl files.

    This class replaces those paths with paths RELATIVE to the project folder
    (e.g. mlruns/1/...), so everything works from any folder, as long as MLflow
    (the script or `mlflow ui`) is started from inside the project folder.
    """

    # (table, column, primary key) where MLflow stores artifact paths
    COLUMNS = [
        ("experiments", "artifact_location", "experiment_id"),
        ("runs", "artifact_uri", "run_uuid"),
        ("logged_models", "artifact_location", "model_id"),
    ]
    # LOCAL path up to ".../mlruns" (C:/..., /home/..., ./..., with or without "file:");
    # remote paths (s3://, http://, mlflow-artifacts:/) are left unchanged
    PATTERN = re.compile(r"^(?:file:(?:///)?)?(?:[a-z]:|\.|[\\/])(?![\\/]{2}).*?[\\/]mlruns(?=[\\/]|$)", re.IGNORECASE)

    def __init__(self, database="mlflow.db", artifacts_dir="mlruns"):
        self.database = Path(database)
        self.artifacts_dir = Path(artifacts_dir)

    def relative(self, path):
        """'file:c:/Users/x/project/mlruns/1/abc' -> 'mlruns/1/abc' (other paths unchanged)."""
        if not path or not self.PATTERN.match(path):
            return path
        return self.PATTERN.sub("mlruns", path).replace("\\", "/")

    def fix_database(self):
        """Converts every artifact path in mlflow.db to a relative path."""
        changed = 0
        with sqlite3.connect(self.database) as con:
            for table, column, key in self.COLUMNS:
                for id_, path in con.execute(f"SELECT {key}, {column} FROM {table}").fetchall():
                    new = self.relative(path)
                    if new != path:
                        con.execute(f"UPDATE {table} SET {column} = ? WHERE {key} = ?", (new, id_))
                        changed += 1
        return changed

    def fix_mlmodel_files(self):
        """Makes the (informative) 'artifact_path' field of the MLmodel files relative."""
        changed = 0
        for file in self.artifacts_dir.rglob("MLmodel"):
            text = file.read_text(encoding="utf-8")
            new = re.sub(r"^(artifact_path: )(.+)$", lambda m: m.group(1) + self.relative(m.group(2)),
                         text, flags=re.MULTILINE)
            if new != text:
                file.write_text(new, encoding="utf-8")
                changed += 1
        return changed

    def fix(self, verbose=True):
        """Fixes the database and the MLmodel files. Safe to run many times."""
        if not self.database.exists():
            return 0
        n_database, n_mlmodel = self.fix_database(), self.fix_mlmodel_files()
        if verbose and (n_database or n_mlmodel):
            print(f"Portability: {n_database} database paths and {n_mlmodel} MLmodel files made relative.")
        return n_database + n_mlmodel

    def absolute_paths(self):
        """Lists what still has an absolute path (should be empty)."""
        with sqlite3.connect(self.database) as con:
            return [(t, c, v) for t, c, k in self.COLUMNS
                    for (v,) in con.execute(f"SELECT {c} FROM {t}") if v and self.PATTERN.match(v)]
