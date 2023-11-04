from typing import List, Tuple, Union, Optional
import numpy as np
from torch.nn import Module
from transaction_anomaly_detection.models.tools.early_stopping.base import (
    EarlyStopperBase,
)


class StandardStopper(EarlyStopperBase):
    def __init__(
        self,
        patience: int,
        delta_threshold: float,
        max_n_epochs: Optional[Union[int, float]] = np.inf,
    ):
        super().__init__(max_n_epochs=max_n_epochs)
        self._patience = patience
        self._delta_threshold = delta_threshold
        self._n_epochs_without_improvement: int = 0

    @property
    def patience(self) -> int:
        return self._patience

    @property
    def delta_threshold(self) -> float:
        return self._delta_threshold

    def update(self, metric: float, model: Module):
        self._update_n_epochs_without_improvement(latest_metric=metric)
        super().update(metric=metric, model=model)

    def update_stop_state(self):
        if not self._stop:
            self._stop = self.stopping_condition_met(
                max_n_epochs=self._max_n_epochs,
                patience=self._patience,
                n_epochs_ellapsed=self._n_epochs_ellapsed,
                n_epochs_without_improvement=self._n_epochs_without_improvement,
            )

    @classmethod
    def stopping_condition_met(
        cls,
        max_n_epochs: Union[int, float],
        patience: int,
        n_epochs_ellapsed: int,
        n_epochs_without_improvement: int,
    ) -> bool:
        max_n_epochs_exceeded = super().stopping_condition_met(
            n_epochs_ellapsed=n_epochs_ellapsed, max_n_epochs=max_n_epochs
        )
        if max_n_epochs_exceeded:
            return True
        if n_epochs_ellapsed < patience:
            return False
        return n_epochs_without_improvement >= patience

    def _update_n_epochs_without_improvement(self, latest_metric: float):
        n_epochs_without_improvement_updated = (
            self._get_n_epochs_without_improvement_updated(
                n_epochs_without_improvement=self._n_epochs_without_improvement,
                latest_metric=latest_metric,
                best_metric=self._best_metric,
                delta_threshold=self._delta_threshold,
            )
        )
        self._n_epochs_without_improvement = n_epochs_without_improvement_updated

    @staticmethod
    def _get_n_epochs_without_improvement_updated(
        n_epochs_without_improvement: int,
        latest_metric: float,
        best_metric: float,
        delta_threshold: float,
    ) -> int:
        if best_metric - latest_metric > delta_threshold:
            return 0
        else:
            return 1 + n_epochs_without_improvement
