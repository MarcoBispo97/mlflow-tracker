from dataclasses import dataclass, field

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DataSplit:
    """Holds the train/test split and the configuration used to create it.
    Every other class receives this object instead of 4 loose variables."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    config: dict = field(default_factory=dict)  # e.g. test_size, random_state, stratify

    @classmethod
    def split(cls, df, features, target, test_size=0.40, random_state=42, stratify=True):
        """Fixed holdout split, stratified by the target.
        - test_size: fraction for the test set (0.40 -> 60% train / 40% test)
        - random_state: makes the split reproducible
        - stratify: keeps the same class proportion in train and test"""
        X, y = df[features], df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y if stratify else None,
        )
        config = {"test_size": test_size, "random_state": random_state, "stratify": stratify}
        return cls(X_train, X_test, y_train, y_test, config=config)

    @property
    def features(self):
        return list(self.X_train.columns)

    @property
    def positive_weight(self):
        """negatives / positives in the training set (weight for the minority class)."""
        return (self.y_train == 0).sum() / (self.y_train == 1).sum()

    def summary(self):
        """Size and share of positives in each part."""
        return pd.DataFrame({
            "rows": [len(self.X_train), len(self.X_test)],
            "positives (%)": [round(100 * self.y_train.mean(), 1), round(100 * self.y_test.mean(), 1)],
        }, index=["train", "test"])
