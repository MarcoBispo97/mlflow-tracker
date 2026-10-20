"""BaseTracker: the MLflow work shared by ClassificationTracker and RegressionTracker.

A tracker trains scikit-learn models and saves everything about each one to MLflow
(model, metrics, charts, learning curve, SHAP), never the same model twice. The child
classes only say how a model is graded: its metrics and its charts.
"""

import hashlib
import re
import sqlite3
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from mlflow.models import infer_signature
from sklearn.base import clone

sns.set_theme(style="whitegrid")  # every chart gets a white background with a grid


class BaseTracker:
    """Trains models and saves each one to MLflow. Use a child class, not this one.

    Args:
        experiment_name: Name of the MLflow experiment (created if it does not exist).
        X_train: Training features (numeric).
        X_test: Test features.
        y_train: Training target.
        y_test: Test target.
        data_source: Where the data came from, e.g. ``"data/titanic.csv"``. Saved with the
            datasets of each run; use a relative path so it works on any machine.
    """

    METRICS = []  # names of the metrics returned by compute_metrics()
    BEST = None  # metric used to rank the models
    LOWER_IS_BETTER = []  # metrics where a smaller value is better (errors)

    def __init__(self, experiment_name, X_train, X_test, y_train, y_test, data_source=None):
        self.experiment = mlflow.set_experiment(experiment_name)
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.data_source = data_source
        make_paths_relative()

    # ---------- what each child class must define ----------
    def compute_metrics(self, model, X, y):
        """Grades a fitted model on the rows ``X`` with the true values ``y``.

        Returns:
            dict[str, float]: One value for each name in ``METRICS``.
        """
        raise NotImplementedError

    def log_charts(self, model):
        """Logs the charts of a fitted model on the test set."""
        raise NotImplementedError

    def predict_for_shap(self, model, X):
        """The number SHAP explains: a probability (classification) or a value (regression)."""
        raise NotImplementedError

    # ---------- training ----------
    def run_all(self, models, top_n=5):
        """Runs every model of a ``{name: model}`` dictionary, then compares the top models.

        Args:
            models: ``{name: model}`` dictionary.
            top_n: Number of best models compared at the end (see ``log_comparison``).

        Returns:
            pandas.DataFrame: One row per model with its test metrics.
        """
        rows = [{"model": name, **self.run(name, model)} for name, model in models.items()]
        self.log_comparison(top_n)
        return pd.DataFrame(rows)

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
            return {m: previous.iloc[0][f"metrics.{m}"] for m in self.METRICS}

        with mlflow.start_run(run_name=self.run_name(name)):
            mlflow.set_tag("model", name)
            self.log_datasets()
            start = time.perf_counter()
            model.fit(self.X_train, self.y_train)
            mlflow.log_metric("train_seconds", time.perf_counter() - start)
            metrics = self.compute_metrics(model, self.X_test, self.y_test)
            mlflow.log_metrics(metrics)
            self.log_charts(model)
            self.log_learning_curve(model)
            self.log_hyperparameters(name, model)
            self.log_learned_parameters(model)

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
            self.log_shap(model)

        make_paths_relative()
        print(f"[{name}] saved")
        return metrics

    def run_name(self, name):
        """Project + model + date and time, e.g. ``titanic_lightgbm_20261019_201530``."""
        return f"{slug(self.experiment.name)}_{slug(name)}_{datetime.now():%Y%m%d_%H%M%S}"

    def log_datasets(self):
        """Logs the training and test data in the "Datasets" field of the current run.

        MLflow saves the schema, a profile and a hash (digest) of each dataset, plus
        ``data_source``. The rows themselves stay in the CSV file.
        """
        target = self.y_train.name or "target"
        for context, X, y in [
            ("training", self.X_train, self.y_train),
            ("testing", self.X_test, self.y_test),
        ]:
            dataset = mlflow.data.from_pandas(
                X.assign(**{target: y}),
                name=f"{slug(self.experiment.name)}_{context}",
                targets=target,
                source=self.data_source,
            )
            mlflow.log_input(dataset, context=context)

    @staticmethod
    def hyperparameters(model):
        """Every hyperparameter of the model, e.g. ``{"max_depth": 4}``.

        For a pipeline the names are ``<step>__<param>`` (e.g. ``xgbclassifier__max_depth``).
        The steps themselves (whole estimators and the list of steps) are left out.
        """
        return {
            name: value
            for name, value in model.get_params().items()
            if not hasattr(value, "get_params") and not isinstance(value, list)
        }

    def log_hyperparameters(self, name, model):
        """Saves every hyperparameter the model used, in two places of the current run.

        - Parameters: one row per hyperparameter, plus the pipeline steps and the data sizes.
        - ``params/hyperparameters.json``: the same values in one JSON file, easy to read
          or to rebuild the model from.
        """
        steps = model.steps if hasattr(model, "steps") else [("model", model)]
        pipeline = [type(step).__name__ for _, step in steps]
        hyperparameters = self.hyperparameters(model)
        mlflow.log_params(
            {
                "pipeline": " -> ".join(pipeline),
                "features": ", ".join(self.X_train.columns),
                "n_train": len(self.X_train),
                "n_test": len(self.X_test),
                **hyperparameters,
            }
        )
        simple = (str, int, float, bool, type(None))  # JSON types; anything else becomes text
        mlflow.log_dict(
            {
                "model": name,
                "pipeline": pipeline,
                "hyperparameters": {
                    k: v if isinstance(v, simple) else str(v) for k, v in hyperparameters.items()
                },
            },
            "params/hyperparameters.json",
        )

    @staticmethod
    def learned_parameters(model):
        """What the model learned from the data during ``fit``: its parameters.

        scikit-learn names them with a trailing underscore, e.g. ``coef_`` and ``intercept_``
        of a linear model, ``feature_importances_`` of a tree or ``mean_`` of a scaler.
        For a pipeline the names are ``<step>__<attribute>``. Big objects (the trees
        themselves, the stored training rows) are left out.
        """
        steps = model.steps if hasattr(model, "steps") else [(type(model).__name__.lower(), model)]
        learned = {}
        for step_name, step in steps:
            for attribute in dir(step):
                if not attribute.endswith("_") or attribute.startswith("_"):
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")  # deprecated attributes
                        value = getattr(step, attribute)
                except Exception:  # some attributes only exist for some settings
                    continue
                if isinstance(value, np.ndarray) and value.dtype.kind in "biuf":
                    if value.size > 1000:
                        continue
                    value = value.round(6).tolist()
                elif isinstance(value, np.generic):
                    value = value.item()
                elif not isinstance(value, (str, int, float, bool)):
                    continue
                learned[f"{step_name}__{attribute}"] = value
        return learned

    def log_learned_parameters(self, model):
        """Saves what the fitted model learned in ``params/learned_parameters.json``.

        The short values (up to 250 characters) also go to Parameters as ``learned__<name>``.
        """
        learned = self.learned_parameters(model)
        mlflow.log_dict(learned, "params/learned_parameters.json")
        mlflow.log_params({f"learned__{k}": v for k, v in learned.items() if len(str(v)) <= 250})

    def fingerprint(self, name, model):
        """Short hash of everything that defines a model: name, hyperparameters and data.

        If any of them changes, the fingerprint changes and the model is trained again.
        """
        text = name + str(model.get_params())
        for part in (self.X_train, self.X_test, self.y_train, self.y_test):
            text += part.to_csv()
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    # ---------- learning curve ----------
    def log_learning_curve(self, model, steps=(0.2, 0.4, 0.6, 0.8, 1.0)):
        """Shows how the model learns: trains copies on 20%, 40%, ... 100% of the training set.

        For each size, the metrics on the rows it learned from (train) and on the test
        set are saved as MLflow metrics with ``step`` = number of rows
        (``learning_train_<metric>`` and ``learning_test_<metric>``), and drawn in
        ``charts/learning_curve.png``. A big gap between train and test means overfitting.
        """
        rows = []
        for fraction in steps:
            n = int(len(self.X_train) * fraction)
            X, y = self.X_train.iloc[:n], self.y_train.iloc[:n]  # already shuffled by the split
            copy = clone(model).fit(X, y)
            for split, scores in [
                ("train", self.compute_metrics(copy, X, y)),
                ("test", self.compute_metrics(copy, self.X_test, self.y_test)),
            ]:
                mlflow.log_metrics({f"learning_{split}_{m}": v for m, v in scores.items()}, step=n)
                rows += [
                    {"rows": n, "split": split, "metric": m, "value": v} for m, v in scores.items()
                ]
        log_chart(plot_curves(pd.DataFrame(rows), hue="split"), "charts/learning_curve.png")

    def plot_learning_curves(self, n=5):
        """Test-set learning curves of the top n models together, one panel per metric.

        Returns:
            matplotlib.figure.Figure: The chart (shown automatically in a notebook).
        """
        top = self.top_models(n)
        client = mlflow.MlflowClient()
        rows = []
        for name, run_id in zip(top["model"], top["run_id"], strict=True):
            for metric in self.METRICS:
                for point in client.get_metric_history(run_id, f"learning_test_{metric}"):
                    rows.append(
                        {"rows": point.step, "model": name, "metric": metric, "value": point.value}
                    )
        fig = plot_curves(pd.DataFrame(rows), hue="model")
        plt.close(fig)  # returned, not drawn twice
        return fig

    # ---------- SHAP ----------
    def log_shap(self, model):
        """Explains the model with SHAP and saves two charts and a CSV.

        Works with any model, because SHAP only calls ``predict_for_shap``. The mean |SHAP|
        of all features goes in one CSV (not one metric per feature, which would draw one
        chart per feature in the MLflow UI); the comparison run reads it to compare models.
        """

        def predict(X):  # SHAP passes a numpy array; pipelines need the column names
            return self.predict_for_shap(model, pd.DataFrame(X, columns=self.X_train.columns))

        background = self.X_train.sample(min(100, len(self.X_train)), random_state=42)
        sample = self.X_test.sample(min(200, len(self.X_test)), random_state=42)
        explanation = shap.Explainer(predict, background)(sample, silent=True)

        importance = pd.DataFrame(
            {
                "feature": self.X_train.columns,
                "mean_abs_shap": np.abs(explanation.values).mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False)
        mlflow.log_text(importance.to_csv(index=False), "shap/importance.csv")

        fig, ax = plt.subplots(figsize=(6, 3))
        sns.barplot(data=importance, x="mean_abs_shap", y="feature", ax=ax)
        ax.set(title="Feature importance (mean |SHAP|)", xlabel="mean |SHAP|", ylabel="")
        log_chart(fig, "shap/importance.png")

        # each dot is one row: position = effect on the prediction, color = feature value
        plt.figure()
        shap.plots.beeswarm(explanation, show=False)
        log_chart(plt.gcf(), "shap/beeswarm.png")

    # ---------- best models ----------
    def top_models(self, n=5, metric=None):
        """Finds the n best models of the experiment (best run of each model name).

        Args:
            n: Number of models.
            metric: Metric used to rank them. Defaults to ``BEST``.

        Returns:
            pandas.DataFrame: ``model``, the metrics, ``run_id`` and ``model_id``, best first.
        """
        metric = metric or self.BEST
        order = "ASC" if metric in self.LOWER_IS_BETTER else "DESC"
        runs = mlflow.search_runs(
            [self.experiment.experiment_id], order_by=[f"metrics.{metric} {order}"]
        )
        if not runs.empty:
            runs = runs[runs["tags.fingerprint"].notna()]  # only training runs have a fingerprint
        if runs.empty:
            return pd.DataFrame(columns=["model", *self.METRICS, "run_id", "model_id"])
        table = pd.DataFrame({"model": runs["tags.model"]})
        for m in self.METRICS:
            table[m] = runs[f"metrics.{m}"]
        table["run_id"] = runs["run_id"]
        table["model_id"] = runs["tags.model_id"]
        return table.drop_duplicates("model").head(n).reset_index(drop=True)

    def load_best_model(self, metric=None):
        """Loads the best model of the experiment, ready for ``predict``.

        Returns:
            tuple: The fitted model and its row of ``top_models()``.
        """
        best = self.top_models(n=1, metric=metric).iloc[0]
        return mlflow.sklearn.load_model(f"models:/{best['model_id']}"), best

    def plot_overview(self, n=5, metric=None):
        """Bar chart that compares the metrics of the top n models.

        Returns:
            matplotlib.figure.Figure: The chart (shown automatically in a notebook).
        """
        top = self.top_models(n, metric)
        scores = top.melt(
            id_vars="model", value_vars=self.METRICS, var_name="metric", value_name="score"
        )
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=scores, x="metric", y="score", hue="model", ax=ax)
        ax.set(title=f"Top {len(top)} models by {metric or self.BEST}", xlabel="")
        ax.legend(fontsize="small")
        fig.tight_layout()
        plt.close(fig)  # returned, not drawn twice
        return fig

    def comparison_charts(self, n=5):
        """Every chart that compares the top n models, built from what MLflow already saved.

        Returns:
            dict[str, matplotlib.figure.Figure]: ``{file name: chart}``.
        """
        top = self.top_models(n)
        client = mlflow.MlflowClient()
        saved = pd.DataFrame(
            [client.get_run(run_id).data.metrics for run_id in top["run_id"]], index=top["model"]
        )

        # metrics heatmap: the color goes from worst (light) to best (dark) in each column
        table = top.set_index("model")[self.METRICS]
        colors = ((table - table.min()) / (table.max() - table.min())).fillna(1)
        for m in self.LOWER_IS_BETTER:
            colors[m] = 1 - colors[m]
        fig_heatmap, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(colors, annot=table, fmt=".3f", cmap="Greens", cbar=False, ax=ax)
        ax.set(title="Test metrics (darker = better)", xlabel="", ylabel="")

        # ranking by the main metric
        fig_ranking, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(data=top, x=self.BEST, y="model", ax=ax)
        ax.bar_label(ax.containers[0], fmt="%.3f")
        ax.set(title=f"Ranking by {self.BEST}", ylabel="")

        # training time
        fig_time, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(x=saved["train_seconds"], y=saved.index, ax=ax)
        ax.bar_label(ax.containers[0], fmt="%.2f s")
        ax.set(title="Training time", xlabel="seconds", ylabel="")

        # which features each model relies on (read from the shap/importance.csv of each run)
        importance = pd.DataFrame(
            {
                name: pd.read_csv(
                    mlflow.artifacts.download_artifacts(
                        run_id=run_id, artifact_path="shap/importance.csv"
                    )
                ).set_index("feature")["mean_abs_shap"]
                for name, run_id in zip(top["model"], top["run_id"], strict=True)
            }
        ).T
        fig_shap, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(importance, annot=True, fmt=".3f", cmap="Reds", cbar=False, ax=ax)
        ax.set(title="Feature importance by model (mean |SHAP|)", xlabel="", ylabel="")

        # the same values as bars: every feature of every model in one chart
        bars = importance.reset_index(names="model").melt(
            id_vars="model", var_name="feature", value_name="mean |SHAP|"
        )
        fig_shap_bars, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=bars, x="mean |SHAP|", y="feature", hue="model", ax=ax)
        ax.set(title="SHAP: feature importance of the top models", ylabel="")
        ax.legend(fontsize="small")

        for fig in (fig_heatmap, fig_ranking, fig_time, fig_shap, fig_shap_bars):
            fig.tight_layout()
            plt.close(fig)  # returned, not drawn twice
        return {
            "top_models.png": self.plot_overview(n),
            "metrics_heatmap.png": fig_heatmap,
            "ranking.png": fig_ranking,
            "learning_curves.png": self.plot_learning_curves(n),
            "train_time.png": fig_time,
            "shap_importance.png": fig_shap,
            "shap_comparison.png": fig_shap_bars,
        }

    def log_comparison(self, n=5):
        """Saves a "Top n comparison" run with the comparison charts and the ranking.

        A new comparison run is created only when the top n models changed, so
        running the same models again does not create duplicates.
        """
        top = self.top_models(n)
        if top.empty:
            return
        top_ids = ",".join(top["run_id"])
        previous = mlflow.search_runs(
            [self.experiment.experiment_id], filter_string=f"tags.top_run_ids = '{top_ids}'"
        )
        if not previous.empty:
            print(f"Top {n} did not change -> comparison already saved")
            return

        with mlflow.start_run(run_name=self.run_name(f"top {n} comparison")):
            mlflow.set_tag("top_run_ids", top_ids)
            for file, fig in self.comparison_charts(n).items():
                log_chart(fig, f"comparison/{file}")
            log_dataframe(top, "comparison/top_models", f"Top {n} models by {self.BEST}")
        make_paths_relative()
        print(f"Top {n} comparison saved (best: {top.loc[0, 'model']})")


def slug(text):
    """Lowercase words joined by underscores: ``"K-Nearest Neighbors"`` -> ``"k_nearest_neighbors"``."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def plot_curves(data, hue):
    """Line chart of ``value`` by number of training ``rows``, one panel per ``metric``."""
    grid = sns.relplot(
        data=data,
        x="rows",
        y="value",
        hue=hue,
        col="metric",
        col_wrap=3,
        kind="line",
        marker="o",
        height=3,
        facet_kws={"sharey": False},
    )
    grid.set_titles("{col_name}")
    grid.set_axis_labels("Training rows", "")
    sns.move_legend(grid, "upper center", bbox_to_anchor=(0.5, 0), ncol=5, title=None)
    grid.figure.suptitle("Learning curve: how the model learns with more data", y=1.05)
    return grid.figure


TABLE_CSS = """
body { font-family: Segoe UI, Arial, sans-serif; font-size: 13px; margin: 12px; }
table { border-collapse: collapse; }
th, td { padding: 4px 12px; text-align: right; }
thead th { border-bottom: 2px solid #333; }
tbody tr:nth-child(odd) { background: #f5f5f5; }
tbody tr:hover { background: #e1efff; }
"""


def log_dataframe(df, path, title):
    """Saves a DataFrame to the current MLflow run as a pandas-style table and a CSV.

    The MLflow UI shows ``<path>.html`` as a table (like a DataFrame in a notebook);
    ``<path>.csv`` is there to download it.
    """
    table = df.to_html(index=False, border=0, na_rep="", float_format="{:.3f}".format)
    mlflow.log_text(f"<style>{TABLE_CSS}</style><h3>{title}</h3>{table}", f"{path}.html")
    mlflow.log_text(df.to_csv(index=False), f"{path}.csv")


def log_chart(fig, path):
    """Saves a matplotlib figure to the current MLflow run and closes it."""
    fig.tight_layout()
    mlflow.log_figure(fig, path, save_kwargs={"bbox_inches": "tight"})  # nothing cut off
    plt.close(fig)


def make_paths_relative(database="mlflow.db"):
    """Makes the artifact paths saved in the MLflow database relative.

    MLflow saves absolute paths (e.g. ``file:C:/Users/me/project/mlruns/1/...``), which
    break when the project is moved to another folder or computer. This turns them into
    ``mlruns/1/...``, so MLflow (the notebooks or ``mlflow ui``) must be started from the
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
