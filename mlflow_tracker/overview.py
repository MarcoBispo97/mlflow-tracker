from datetime import datetime

import mlflow
import mlflow.sklearn
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from sklearn.metrics import roc_auc_score, roc_curve


class Overview:
    """Overview page of the top N models (by one metric), saved to MLflow and to disk.

    Steps of generate():
      1. find_top       -> the N best training runs of the experiment (no duplicates)
      2. *_chart        -> interactive metric bars and ROC curves
      3. build_page     -> ranking + charts in a single HTML page
      4. save to MLflow -> run "Overview Top N" > Artifacts > overview/overview.html
         and to disk    -> overview.html (opens in the browser)
    """

    METRICS = ["accuracy", "precision", "recall", "f1"]
    CSS = """
    body  { font-family: Segoe UI, Arial, sans-serif; margin: 0; background: #f4f6f9; color: #1f2937; }
    header{ background: #0b5cad; color: white; padding: 20px 32px; }
    header h1 { margin: 0 0 4px; font-size: 24px; }
    main  { padding: 24px 32px; display: grid; gap: 20px; }
    .card { background: white; border-radius: 10px; padding: 16px 20px; box-shadow: 0 1px 4px #0002; }
    .best { font-size: 18px; } .best b { color: #0b5cad; }
    table { border-collapse: collapse; width: 100%; }
    th, td { padding: 8px 10px; border-bottom: 1px solid #e5e7eb; text-align: center; }
    th { background: #eef2f7; }
    """

    def __init__(self, experiment_id, data, n=5, metric="f1", file="overview.html"):
        self.experiment_id = experiment_id
        self.data = data
        self.n = n
        self.metric = metric
        self.file = file

    # ---------- 1. data ----------
    def find_top(self):
        """The n best training runs, best first, each model only once."""
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string="tags.kind = 'training'",
            order_by=[f"metrics.{self.metric} DESC"],
        )
        if runs.empty:
            return pd.DataFrame()
        runs = runs.sort_values(
            [f"metrics.{self.metric}", "start_time"], ascending=[False, False]
        ).drop_duplicates("tags.fingerprint").head(self.n)

        table = pd.DataFrame({
            "model": runs["tags.model_name"].values,
            "run_id": runs["run_id"].values,
            "model_id": runs["tags.model_id"].values,
        })
        for m in self.METRICS + ["roc_auc"]:
            table[m] = runs[f"metrics.{m}"].values
        return table

    # ---------- 2. charts ----------
    def metrics_chart(self, table):
        """Grouped bars: one group per metric, one bar per model."""
        fig = go.Figure()
        for _, row in table.iterrows():
            values = row[self.METRICS].astype(float)
            fig.add_bar(name=row["model"], x=self.METRICS, y=values,
                        text=values.round(2), textposition="outside", textfont_size=10)
        fig.update_layout(barmode="group", title="Metrics comparison (test set)",
                          yaxis_range=[0, 1.1], height=500, legend=dict(orientation="h", y=-0.15))
        return fig

    def roc_chart(self, table):
        """ROC curve of each model (loads the model saved in MLflow)."""
        X_test, y_test = self.data.X_test, self.data.y_test
        fig = go.Figure()
        for _, row in table.iterrows():
            model = mlflow.sklearn.load_model(f"models:/{row['model_id']}")
            if hasattr(model, "predict_proba"):
                score = model.predict_proba(X_test)[:, 1]
            else:  # e.g. SVM without probability=True
                score = model.decision_function(X_test)
            fpr, tpr, _ = roc_curve(y_test, score)
            auc = roc_auc_score(y_test, score)
            fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{row['model']} (AUC={auc:.3f})")
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                        line=dict(dash="dash", color="gray"))
        fig.update_layout(title="ROC curves", xaxis_title="False positive rate",
                          yaxis_title="True positive rate", height=600,
                          legend=dict(orientation="h", y=-0.2))
        return fig

    # ---------- 3. page ----------
    def build_page(self, table, fig_metrics, fig_roc):
        """Ranking + charts in a single HTML page (works offline)."""
        best = table.iloc[0]
        columns = self.METRICS + ["roc_auc"]
        ranking = table.drop(columns=["run_id", "model_id"]).copy()
        ranking.insert(0, "#", range(1, len(ranking) + 1))
        ranking_html = (
            ranking.style
            .format({m: "{:.3f}" for m in columns})
            .highlight_max(subset=columns, props="background:#d1fae5; font-weight:bold")
            .hide(axis="index")
            .to_html()
        )
        split_info = ", ".join(f"{k}={v}" for k, v in self.data.config.items()) or "holdout"
        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Models overview</title>
<style>{self.CSS}</style><script>{get_plotlyjs()}</script></head>
<body>
<header>
  <h1>Overview - Top {len(table)} models by {self.metric.upper()}</h1>
  <div>Updated on {datetime.now():%Y-%m-%d %H:%M:%S} · Evaluation: {split_info}</div>
</header>
<main>
  <div class="card best">Best model: <b>{best['model']}</b> ·
    {self.metric.upper()} = <b>{best[self.metric]:.3f}</b> · ROC AUC = <b>{best['roc_auc']:.3f}</b></div>
  <div class="card"><h2>Ranking</h2>{ranking_html}
    <p>Green = best value in the column.</p></div>
  <div class="card">{fig_metrics.to_html(full_html=False, include_plotlyjs=False)}</div>
  <div class="card">{fig_roc.to_html(full_html=False, include_plotlyjs=False)}</div>
</main></body></html>"""

    # ---------- 4. save ----------
    def _existing_run(self, top_key):
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=f"tags.kind = 'overview' and tags.top_run_ids = '{top_key}'",
            max_results=1,
        )
        return None if runs.empty else runs.iloc[0]["run_id"]

    def _save_to_mlflow(self, table, html, top_key):
        with mlflow.start_run(experiment_id=self.experiment_id,
                              run_name=f"Overview Top {self.n} - {self.metric.upper()}"):
            mlflow.set_tags({"kind": "overview", "top_run_ids": top_key})
            mlflow.log_text(html, "overview/overview.html")
            mlflow.log_table(table, "overview/top_models.json")
        # the ranking also goes to the experiment description
        columns = self.METRICS + ["roc_auc"]
        lines = [f"## Top {self.n} by {self.metric.upper()}",
                 "Full page: run **Overview** > Artifacts > `overview/overview.html`", "",
                 "| # | model | " + " | ".join(columns) + " |",
                 "|---" * (len(columns) + 2) + "|"]
        for i, r in enumerate(table.itertuples(index=False), 1):
            lines.append(f"| {i} | {r.model} | " + " | ".join(f"{getattr(r, m):.3f}" for m in columns) + " |")
        mlflow.MlflowClient().set_experiment_tag(self.experiment_id, "mlflow.note.content", "\n".join(lines))

    def generate(self, show=False):
        """Builds the top N page and saves it (a new MLflow run only if the top N changed)."""
        table = self.find_top()
        if table.empty:
            print("No training runs found.")
            return table

        fig_metrics = self.metrics_chart(table)
        fig_roc = self.roc_chart(table)
        html = self.build_page(table, fig_metrics, fig_roc)

        with open(self.file, "w", encoding="utf-8") as f:  # local copy, always up to date
            f.write(html)
        print(f"Overview page saved to: {self.file}")

        top_key = ",".join(table["run_id"])
        existing = self._existing_run(top_key)
        if existing:
            print(f"Top {self.n} did not change -> overview already in MLflow (run {existing}).")
        else:
            self._save_to_mlflow(table, html, top_key)
            print("Overview saved to MLflow.")

        if show:
            fig_metrics.show()
            fig_roc.show()
        return table.drop(columns=["model_id"])
