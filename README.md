# MLflow Tracker

#### Video Demo: <PASTE_YOUR_YOUTUBE_URL_HERE>

#### Description:

**MLflow Tracker** is a small Python package that trains binary classification models with
scikit-learn and saves *everything* about each model to [MLflow](https://mlflow.org): the trained
model, a `.pkl` copy, metrics, interactive charts, SHAP explanations, the features used and how the
model was evaluated. It never saves the same model twice, it builds an overview page that compares
the best models, and the whole project can be copied to another computer without breaking MLflow.

The package is generic: it works with any scikit-learn compatible model or `Pipeline`. To show it
working, the repository includes `titanic.py`, which trains five classifiers on the classic
**Titanic** dataset (who survived the sinking?) with a single command.

---

## Quick start

Requires **Python 3.12**.

```bash
git clone https://github.com/MarcoBispo97/mlflow-tracker.git
cd mlflow-tracker

python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python titanic.py                  # add --show to open every chart in the browser
```

The script prints the train/test split, trains the five models, prints the top 5 ranking and
predicts the survival of four new passengers with the best model:

```
Best model: titanic_xgboost_20261016_203209 (F1 = 0.753)
 Pclass    Sex  Age  survives  probability
      1 female 29.0      True        0.964
      3   male 22.0     False        0.075
      2   male  NaN     False        0.148
      3 female  5.0      True        0.782
```

Then open the MLflow UI **from inside the project folder** and go to the `titanic` experiment:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# open http://127.0.0.1:5000
```

The overview page is also saved as `overview.html`, which you can open directly in the browser.

Run `python titanic.py` a second time and nothing is trained again: every model prints
`nothing changed -> already saved as ...`. Change a hyperparameter in `titanic.py` and only that
model gets a new run.

---

## Using it with your own data

```python
import mlflow
from mlflow_tracker import DataSplit, MLflowTracker, PlotlyCharts

data = DataSplit.split(df, features=["a", "b", "c"], target="label", test_size=0.3)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
tracker = MLflowTracker("my_experiment", data, PlotlyCharts(labels=("No", "Yes")))
tracker.run_all({"Random Forest": RandomForestClassifier(), "SVM": my_svm_pipeline})

tracker.save_overview(n=5, metric="f1")
model, best_run = tracker.load_best_model(metric="f1")
```

The only requirements are a binary target (0/1) and numeric features.

---

## Files

| File | What it does |
|---|---|
| `mlflow_tracker/data.py` | `DataSplit`: a dataclass that holds `X_train`, `X_test`, `y_train`, `y_test` and the split configuration. `DataSplit.split()` makes a stratified holdout split, and `positive_weight` gives the weight of the minority class, used by models that need it (XGBoost). Every other class receives this one object instead of four loose variables. |
| `mlflow_tracker/tracker.py` | `MLflowTracker`: the core of the project. For each model it computes a fingerprint, skips the model if that fingerprint already exists, and otherwise opens an MLflow run, trains the model and logs the model (with signature and input example), a `.pkl` named after the model and the date/time, the metrics (accuracy, precision, recall, F1, ROC AUC), the confusion matrix, the ROC curve, SHAP charts, the features and the evaluation method. It also finds the top models and loads the best `.pkl` for inference. |
| `mlflow_tracker/charts.py` | `PlotlyCharts`: builds every chart as an interactive Plotly figure: confusion matrix, ROC curve, SHAP importance bars and a SHAP scatter ("beeswarm") plot. The class labels are configurable. |
| `mlflow_tracker/overview.py` | `Overview`: searches the experiment for the N best runs by a metric, removes duplicates, and builds a single HTML page with the ranking (best value of each column in green), a grouped bar chart of the metrics and the ROC curves of all top models. The page is saved in an MLflow run, in the experiment description and on disk. |
| `mlflow_tracker/portability.py` | `MLflowPortability`: rewrites the absolute artifact paths that MLflow stores in `mlflow.db` (and in the `MLmodel` files) as relative paths (`mlruns/...`). |
| `mlflow_tracker/__init__.py` | Exposes the five classes, so they can be imported with `from mlflow_tracker import ...`. |
| `titanic.py` | The example. It reads `data/titanic.csv`, converts `Sex` to 0/1, splits 60% train / 40% test, creates five pipelines (polynomial logistic regression, SVM, decision tree, LightGBM and XGBoost), tracks them, builds the overview and predicts four new passengers. |
| `data/titanic.csv` | The Titanic dataset (891 passengers), kept in the repository so the example works offline. |
| `requirements.txt` | The direct dependencies, with pinned versions. |
| `tests/` | 27 pytest tests: the split, the path conversion (Windows, Linux, remote and relative paths), the fingerprint, the artifacts of a run, skipping repeated models, loading the best `.pkl` and the overview. Each test uses its own temporary MLflow database. |
| `scripts/run_tests.py` | Runs pytest with the project's `.venv`; used by the pre-commit hook, because git hooks do not activate the virtual environment. |
| `requirements-dev.txt` | The development tools (black, ruff, pytest and pre-commit), with pinned versions. |
| `.pre-commit-config.yaml` | Runs black, ruff and pytest before every `git commit`. |
| `pyproject.toml` | Configuration of black (line length 100) and ruff (pycodestyle, pyflakes, import order, bugbear, pyupgrade and Google-style docstrings). |
| `.github/workflows/ci.yml` | GitHub Actions: on every push it checks the formatting and the lint, runs the tests, then runs `titanic.py` twice and checks that the second run skips all 5 models. |

---

## Design decisions

**Never save the same model twice.** Running a notebook or a script many times usually fills MLflow
with identical runs, and then the ranking shows the same model five times. Before training, the
tracker computes a *fingerprint*: a SHA-256 hash of the model name, its class, all its
hyperparameters, the feature names, the content of the train and test data and the split
configuration. If a training run with that fingerprint already exists, the model is skipped. I
chose to hash the data content, not just its shape, because a different row in the training set
must produce a different model. The overview page uses the same idea: a new overview run is
created only when the list of top runs changes.

**A traceable name for each model.** Every model is saved as
`<experiment>_<model>_<date>_<time>` (for example `titanic_xgboost_20261016_203209`), lowercase and
without accents, and the `.pkl` file has the same name. This way a file found on disk can always be
traced back to its run.

**Interactive charts.** Matplotlib images are static. I used Plotly and saved the figures with
`mlflow.log_figure(fig, "x.html")`, so the charts are interactive inside the MLflow UI: you can
hover over a point of the ROC curve or zoom into the SHAP plot.

**Model-agnostic SHAP.** Tree explainers only work with tree models, and each model type needs a
different explainer. I used `shap.Explainer` with an `Independent` masker over a function that
calls `predict_proba`, so the same code explains a logistic regression, an SVM or XGBoost. The
trade-off is speed, so the explanation uses a sample of 200 test rows against a background of 100
training rows. The mean |SHAP| of each feature is also logged as a metric, which makes it possible
to compare feature importance between models in the MLflow UI.

**Small and imbalanced data.** Only 38% of the passengers survived and there are fewer than 900
rows. The models use class weights (`class_weight="balanced"` or XGBoost's `scale_pos_weight`,
computed from the training set) and conservative hyperparameters: shallow trees, few leaves, a low
learning rate and regularization. The split is stratified, so train and test keep the same 38%.

**Preprocessing inside the pipeline.** Missing ages are filled with the median and the features are
standardized inside each `Pipeline`, so these steps learn only from the training set (no data
leakage) and are saved together with the model. The only step outside the pipeline is mapping
`Sex` to 0/1, a fixed rule that is applied the same way to new passengers.

**SQLite and portable paths.** MLflow 3 deprecated the plain folder backend, so runs and metrics go
to `mlflow.db` (SQLite) and artifacts to `mlruns/`. MLflow stores absolute artifact paths such as
`file:C:/Users/me/project/mlruns/...`, which break as soon as the folder is moved or opened on
another computer. `MLflowPortability` rewrites them as `mlruns/...` every time the tracker starts.
This is why `mlflow ui` must be started from inside the project folder.

**Two ways to load a model.** The MLflow format (`mlflow.pyfunc.load_model("models:/<id>")`) is the
standard one and can be served as a REST API with `mlflow models serve`. The `.pkl` is the simplest
way to use the model in any Python script with `joblib.load`. Only load `.pkl` files you created
yourself, because unpickling a file can run code.

---

## Development

The code is formatted with [black](https://black.readthedocs.io), linted with
[ruff](https://docs.astral.sh/ruff/), and every module, class and public function has a
[Google-style docstring](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
(checked by ruff's `D` rules).

```bash
pip install -r requirements.txt -r requirements-dev.txt
black .            # format
ruff check .       # lint, including the docstrings
pytest             # 27 tests
pre-commit install # run all of the above before every git commit
```

The same checks run on GitHub Actions for every push and pull request, followed by an end-to-end
run of the Titanic example.

---

## Limitations

- Binary classification only (precision, recall and F1 are computed for the positive class).
- The evaluation is a single holdout split; cross-validation would give more stable metrics.
- The per-run chart files load Plotly from a CDN, so they need internet inside the MLflow UI
  (`overview.html` embeds Plotly and works offline).

---

## AI usage

<!-- Edit this paragraph so it describes exactly how you used AI tools. -->
I designed and built this project in Jupyter notebooks while studying MLflow. I used Claude
(Anthropic's AI assistant) as a helper: to discuss design ideas, debug problems (such as the
absolute paths stored by MLflow), and, at the end, to help me move the classes from my notebook
into this package, translate the identifiers and comments from Portuguese to English, and draft
this README. I reviewed, ran and tested all of the code.

---

This was my final project for [CS50x](https://cs50.harvard.edu/x/).
