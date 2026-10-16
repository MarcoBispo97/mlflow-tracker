import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


class PlotlyCharts:
    """Interactive charts (Plotly). Every method returns a figure that can be shown
    with fig.show() or saved to MLflow with mlflow.log_figure(fig, "x.html")."""

    def __init__(self, labels=("Negative", "Positive")):
        self.labels = list(labels)

    def confusion_matrix(self, y_true, y_pred, title="Confusion matrix"):
        """Heatmap of the confusion matrix (rows = actual, columns = predicted)."""
        fig = px.imshow(
            confusion_matrix(y_true, y_pred), x=self.labels, y=self.labels, text_auto=True,
            color_continuous_scale="Blues", labels=dict(x="Predicted", y="Actual", color="Count"),
        )
        fig.update_layout(title=title, width=520, height=450)
        return fig

    @staticmethod
    def roc_curve(y_true, y_score, name, title=None):
        """ROC curve of one model, with the random-classifier line."""
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        fig = go.Figure()
        fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.3f})")
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                        line=dict(dash="dash", color="gray"))
        fig.update_layout(title=title or f"ROC curve - {name}", width=620, height=500,
                          xaxis_title="False positive rate", yaxis_title="True positive rate",
                          legend=dict(x=0.45, y=0.05))
        return fig

    @staticmethod
    def shap_importance(importance, name):
        """Horizontal bars: mean |SHAP| of each feature (how much it weighs on predictions)."""
        fig = px.bar(
            importance.sort_values("mean_abs_shap"), x="mean_abs_shap", y="feature", orientation="h",
            text_auto=".3f", labels={"mean_abs_shap": "mean |SHAP|", "feature": "Feature"},
        )
        fig.update_layout(title=f"Feature importance (SHAP) - {name}", width=650, height=380)
        return fig

    @staticmethod
    def shap_scatter(shap_values, X, name):
        """Each dot is one row: position = effect on the prediction, color = feature value
        (blue = low, red = high). Shows the 'direction' of each feature."""
        order = np.argsort(np.abs(shap_values).mean(axis=0))  # least important at the bottom
        rng = np.random.default_rng(42)
        fig = go.Figure()
        for position, j in enumerate(order):
            column = X.iloc[:, j].astype(float)
            color = (column - column.min()) / (column.max() - column.min() or 1)  # 0 to 1 per feature
            fig.add_scatter(
                x=shap_values[:, j], y=position + rng.uniform(-0.25, 0.25, len(column)),
                mode="markers", name=X.columns[j], showlegend=False,
                marker=dict(size=6, color=color, colorscale="RdBu_r", cmin=0, cmax=1, opacity=0.8,
                            showscale=(position == 0),  # a single color bar
                            colorbar=dict(title="feature value", tickvals=[0, 1], ticktext=["low", "high"])),
                customdata=column, hovertemplate=f"{X.columns[j]}=%{{customdata:.3f}}<br>SHAP=%{{x:.3f}}<extra></extra>",
            )
        fig.add_vline(x=0, line_color="gray")
        fig.update_layout(title=f"Effect of each feature (SHAP) - {name}", width=750, height=420,
                          xaxis_title="SHAP value (impact on the prediction)",
                          yaxis=dict(tickvals=list(range(len(order))), ticktext=[X.columns[j] for j in order]))
        return fig
