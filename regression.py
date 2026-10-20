"""RegressionTracker: tracks regression models (numeric target) in MLflow."""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from utils import BaseTracker, log_chart


class RegressionTracker(BaseTracker):
    """Tracks regression models.

    Args:
        experiment_name: Name of the MLflow experiment.
        X_train: Training features.
        X_test: Test features.
        y_train: Training target (numbers).
        y_test: Test target (numbers).
        data_source: Where the data came from, e.g. ``"data/house_prices.csv"``.
    """

    METRICS = ["mae", "rmse", "r2"]
    BEST = "rmse"
    LOWER_IS_BETTER = ["mae", "rmse"]  # errors: smaller is better

    def compute_metrics(self, model, X, y):
        """MAE, RMSE and R² of the model on ``X``."""
        y_pred = model.predict(X)
        return {
            "mae": mean_absolute_error(y, y_pred),
            "rmse": root_mean_squared_error(y, y_pred),
            "r2": r2_score(y, y_pred),
        }

    def log_charts(self, model):
        """Logs 4 charts on the test set: predicted vs. actual and 3 about the errors."""
        y_pred = model.predict(self.X_test)

        fig, ax = plt.subplots(figsize=(5, 5))
        sns.scatterplot(x=self.y_test, y=y_pred, alpha=0.3, ax=ax)
        low, high = self.y_test.min(), self.y_test.max()
        ax.plot([low, high], [low, high], "--", color="gray")  # perfect predictions
        ax.set(title="Predicted vs. actual", xlabel="Actual", ylabel="Predicted")
        log_chart(fig, "charts/predicted_vs_actual.png")

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(self.y_test - y_pred, bins=50, ax=ax)
        ax.axvline(0, color="gray", linestyle="--")
        ax.set(title="Residuals (actual - predicted)", xlabel="Residual")
        log_chart(fig, "charts/residuals.png")

        # a good model has residuals spread evenly around 0, whatever the predicted value
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x=y_pred, y=self.y_test - y_pred, alpha=0.3, ax=ax)
        ax.axhline(0, color="gray", linestyle="--")
        ax.set(title="Residuals vs. predicted", xlabel="Predicted", ylabel="Residual")
        log_chart(fig, "charts/residuals_vs_predicted.png")

        # a good model predicts values with the same distribution as the real ones
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.kdeplot(x=self.y_test, fill=True, label="Actual", ax=ax)
        sns.kdeplot(x=y_pred, fill=True, label="Predicted", ax=ax)
        ax.set(title="Distribution: actual vs. predicted", xlabel="Value")
        ax.legend()
        log_chart(fig, "charts/distribution.png")

    def predict_for_shap(self, model, X):
        """SHAP explains the predicted value."""
        return model.predict(X)
