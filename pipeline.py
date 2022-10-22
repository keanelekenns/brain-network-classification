from __future__ import annotations
from typing import Any
import numpy as np
from utils import plot_points


class Pipeline():
    def __init__(self, steps: list[Any], params: dict[str, Any]) -> None:
        # Attributes for plotting training/testing points
        self.axes_labels = None
        self.a_label = None
        self.b_label = None
        self.train_points = None
        self.train_labels = None
        self.test_points = None
        self.test_labels = None
        self.predicted_test_labels = None
        self.plot_prefix = None

        # Init the pipeline steps with the given parameters
        self.params = params
        self.steps = []
        for step in steps:
            if hasattr(step, "takes_pipeline"):
                if not params.get(step.__name__):
                    self.steps.append(step(pipeline=self))
                else:
                    self.steps.append(step(**params[step.__name__], pipeline=self))
            else:
                if not params.get(step.__name__):
                    self.steps.append(step())
                else:
                    self.steps.append(step(**params[step.__name__]))

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_transformed = X_train
        for step in self.steps:
            step.fit(X_transformed, y_train)
            if step != self.steps[-1]:
                X_transformed = step.transform(X_transformed)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = None
        X_test = X
        for step in self.steps:
            if step != self.steps[-1]:
                X_test = step.transform(X_test)
            else:
                # The last step must be a classifier
                y_pred = step.predict(X_test)

        self.predicted_test_labels = y_pred
        return y_pred


    def is_plottable(self) -> bool:
        return not (self.axes_labels is None or
                    self.a_label is None or
                    self.b_label is None or
                    self.train_points is None or
                    self.train_labels is None or
                    self.test_points is None or
                    self.test_labels is None or
                    self.predicted_test_labels is None or
                    self.plot_prefix is None)

    def add_points(self, points: np.ndarray) -> None:
        if self.train_points is None:
            self.train_points = points
        elif self.test_points is None:
            self.test_points = points
        else:
            # Restart by overwriting the train points
            # (can only hold one set of train and test points at a time)
            self.train_points = points
            self.test_points = None



    def plot(self) -> None:
        plot_points(self.train_points, self.train_labels,
                    f"plots/{self.plot_prefix}-train-labels",
                    axes_labels=self.axes_labels,
                    a_label=self.a_label,
                    b_label=self.b_label)

        plot_points(self.test_points, self.test_labels,
                    f"plots/{self.plot_prefix}-test-labels",
                    axes_labels=self.axes_labels,
                    a_label=self.a_label,
                    b_label=self.b_label)

        plot_points(self.test_points, self.predicted_test_labels,
                    f"plots/{self.plot_prefix}-test-predictions",
                    axes_labels=self.axes_labels,
                    a_label=self.a_label,
                    b_label=self.b_label)
                