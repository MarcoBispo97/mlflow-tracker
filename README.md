# MLflow Tracker

#### Video Demo: <PASTE_YOUR_YOUTUBE_URL_HERE>

#### Description:

**MLflow Tracker** is a single Python file (`mlflow_tracker.py`) that trains binary classification
models with scikit-learn and saves *everything* about each model to [MLflow](https://mlflow.org):
the trained model, metrics, charts (seaborn/matplotlib), SHAP explanations and the features used.
It never saves the same model twice, it draws a chart that compares the best models, and the whole
project can be copied to another computer without breaking MLflow.

It is generic: it works with any scikit-learn model or `Pipeline` that has `predict_proba`. To show it
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

python titanic.py
```

The script prints the train/test split, trains the five models, prints the top 5 ranking and
predicts the survival of four new passengers with the best model:

```
Best model: LightGBM (F1 = 0.745)
 Pclass    Sex  Age  survives  probability
      1 female 29.0      True        0.954
      3   male 22.0     False        0.100
      2   male  NaN     False        0.173
      3 female  5.0      True        0.874
```

Then open the MLflow UI **from inside the project folder** and go to the `titanic` experiment:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# open http://127.0.0.1:5000
```

The chart that compares the top 5 models is saved as `overview.png`.

Run `python titanic.py` a second time and nothing is trained again: every model prints
`nothing changed -> skipped`. Change a hyperparameter in `titanic.py` and only that
model gets a new run.

---

## Using it with your own data

```python
import mlflow
from sklearn.model_selection import train_test_split
from mlflow_tracker import MLflowTracker

X_train, X_test, y_train, y_test = train_test_split(df[["a", "b", "c"]], df["label"], test_size=0.3)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
tracker = MLflowTracker("my_experiment", X_train, X_test, y_train, y_test, labels=("No", "Yes"))
tracker.run_all({"Random Forest": RandomForestClassifier(), "SVM": my_svm_pipeline})

tracker.save_overview(n=5, metric="f1")             # overview.png
model, best_run = tracker.load_best_model(metric="f1")
```

The only requirements are a binary target (0/1), numeric features and a model with `predict_proba`.

---

## Files

| File | What it does |
|---|---|
| `mlflow_tracker.py` | Everything in one file. `MLflowTracker.run()` computes a fingerprint of the model, skips it if it was already saved, and otherwise trains it and logs the model, the metrics (accuracy, precision, recall, F1, ROC AUC), a confusion matrix and a ROC curve, two SHAP charts and the features. `top_models()`, `load_best_model()` and `save_overview()` compare and reuse the saved models. `make_paths_relative()` keeps the project portable. |
| `titanic.py` | The example. It reads `data/titanic.csv`, converts `Sex` to 0/1, splits 60% train / 40% test, creates five pipelines (polynomial logistic regression, SVM, decision tree, LightGBM and XGBoost), tracks them, saves the overview and predicts four new passengers. |
| `data/titanic.csv` | The Titanic dataset (891 passengers), kept in the repository so the example works offline. |
| `tests/test_mlflow_tracker.py` | pytest tests: the fingerprint, the artifacts of a run, skipping repeated models, the ranking, loading the best model and the relative paths. Each test uses its own temporary MLflow database. |
| `requirements.txt` / `requirements-dev.txt` | The dependencies and the development tools (black, ruff, pytest, pre-commit), with pinned versions. |
| `pyproject.toml` | Configuration of black, ruff and pytest. |
| `.pre-commit-config.yaml` | Runs black and ruff before every `git commit`. |
| `.github/workflows/ci.yml` | GitHub Actions: checks the formatting and the lint, runs the tests, then runs `titanic.py` twice and checks that the second run skips all 5 models. |

---

## Design decisions

**Never save the same model twice.** Running a script many times usually fills MLflow with
identical runs. Before training, the tracker computes a *fingerprint*: a SHA-256 hash of the model
name, its hyperparameters and the content of the train and test data. If a run with that
fingerprint already exists, the model is skipped. I hash the data content, not just its shape,
because a different row in the training set must produce a different model.

**Simple charts.** The charts are PNG images made with seaborn and matplotlib (and sklearn's
`RocCurveDisplay` and SHAP's own beeswarm plot), saved with `mlflow.log_figure`. They are easy to
read, easy to change and need no extra JavaScript.

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
another computer. `make_paths_relative()` rewrites them as `mlruns/...` with a plain SQL `REPLACE`.
This is why `mlflow ui` must be started from inside the project folder.

**Loading a model.** `load_best_model()` uses `mlflow.sklearn.load_model("models:/<id>")`; the same
model can be served as a REST API with `mlflow models serve`. Models are saved with cloudpickle
because MLflow's default format (skops) refuses tree models; only load models you created yourself,
because unpickling a file can run code.

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
pytest             # tests
pre-commit install # run black and ruff before every git commit
```

The same checks run on GitHub Actions for every push and pull request, followed by an end-to-end
run of the Titanic example.

---

## Limitations

- Binary classification only (precision, recall and F1 are computed for the positive class).
- The evaluation is a single holdout split; cross-validation would give more stable metrics.
- The model needs `predict_proba` (for an SVM, use `SVC(probability=True)`).

---

## AI usage

<!-- Edit this paragraph so it describes exactly how you used AI tools. -->
I designed and built this project in Jupyter notebooks while studying MLflow. I used Claude
(Anthropic's AI assistant) as a helper: to discuss design ideas, debug problems (such as the
absolute paths stored by MLflow), and, at the end, to help me move the code from my notebook
into this project, later simplify it into a single file, translate the identifiers and comments from Portuguese to English, and draft
this README. I reviewed, ran and tested all of the code.

---

This was my final project for [CS50x](https://cs50.harvard.edu/x/).
