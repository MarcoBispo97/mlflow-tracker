"""Train/test split shared by every other class of the package."""

from dataclasses import dataclass, field

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DataSplit:
    """Holds the train/test split and the configuration used to create it.

    Every other class receives this object instead of four loose variables.

    Attributes:
        X_train: Training features.
        X_test: Test features.
        y_train: Training target (0/1).
        y_test: Test target (0/1).
        config: Split configuration, e.g. ``test_size``, ``random_state`` and ``stratify``.
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    config: dict = field(default_factory=dict)

    @classmethod
    def split(cls, df, features, target, test_size=0.40, random_state=42, stratify=True):
        """Creates a fixed holdout split, stratified by the target.

        Args:
            df: DataFrame with the features and the target.
            features: Names of the feature columns.
            target: Name of the target column (0/1).
            test_size: Fraction of rows for the test set (0.40 means 60% train / 40% test).
            random_state: Seed that makes the split reproducible.
            stratify: If True, train and test keep the same class proportion.

        Returns:
            A new ``DataSplit`` with the four parts and the configuration.
        """
        X, y = df[features], df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if stratify else None,
        )
        config = {"test_size": test_size, "random_state": random_state, "stratify": stratify}
        return cls(X_train, X_test, y_train, y_test, config=config)

    @property
    def features(self):
        """list[str]: Names of the feature columns."""
        return list(self.X_train.columns)

    @property
    def positive_weight(self):
        """float: Negatives / positives in the training set (weight of the minority class)."""
        return (self.y_train == 0).sum() / (self.y_train == 1).sum()

    def summary(self):
        """Summarizes the size and the share of positives of each part.

        Returns:
            A DataFrame with one row for ``train`` and one for ``test``.
        """
        return pd.DataFrame(
            {
                "rows": [len(self.X_train), len(self.X_test)],
                "positives (%)": [
                    round(100 * self.y_train.mean(), 1),
                    round(100 * self.y_test.mean(), 1),
                ],
            },
            index=["train", "test"],
        )
