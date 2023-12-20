from typing import List, Tuple, Union, Optional
import numpy as np
from torch.nn import Module
from transaction_anomaly_detection.models.tools.training.early_stopping.base import (
    EarlyStopperBase,
)


class MonotoneStopper(EarlyStopperBase):
    def __init__(
        self,
        patience: int,
        delta_threshold: float,
        max_n_epochs: Optional[Union[int, float]] = np.inf,
    ):
        super().__init__(max_n_epochs=max_n_epochs)
        self._patience = patience
        self._delta_threshold = delta_threshold
        self._ls_historic_metrics: List[float] = []

    @property
    def patience(self) -> int:
        return self._patience

    @property
    def delta_threshold(self) -> float:
        return self._delta_threshold

    def update(self, metric: float, model: Module):
        self._update_ls_historic_metrics(latest_metric=metric)
        super().update(metric=metric, model=model)

    def update_stop_state(self):
        if not self._stop:
            self._stop = self.stopping_condition_met(
                max_n_epochs=self._max_n_epochs,
                patience=self._patience,
                delta_threshold=self._delta_threshold,
                n_epochs_ellapsed=self._n_epochs_ellapsed,
                ls_historic_metrics=self._ls_historic_metrics,
            )

    @classmethod
    def stopping_condition_met(
        cls,
        max_n_epochs: Union[int, float],
        patience: int,
        delta_threshold: float,
        n_epochs_ellapsed: int,
        ls_historic_metrics: List[float],
    ) -> bool:
        max_n_epochs_exceeded = super().stopping_condition_met(
            n_epochs_ellapsed=n_epochs_ellapsed, max_n_epochs=max_n_epochs
        )
        if max_n_epochs_exceeded:
            return True
        if n_epochs_ellapsed < patience:
            return False
        ls_successive_metrics = cls._get_successive_metrics(
            ls_historic_metrics=ls_historic_metrics, patience=patience
        )
        ls_deltas = cls._get_deltas(ls_successive_metrics=ls_successive_metrics)
        ls_deltas_exceeded_threshold = cls._deltas_exceeded_threshold(
            ls_deltas=ls_deltas, delta_threshold=delta_threshold
        )
        return not any(ls_deltas_exceeded_threshold)

    def _update_ls_historic_metrics(self, latest_metric: float):
        ls_historic_metrics_updated = self._get_ls_historic_metrics_updated(
            latest_metric=latest_metric,
            patience=self._patience,
            ls_historic_metrics_current=self._ls_historic_metrics,
        )
        self._ls_historic_metrics = ls_historic_metrics_updated

    @staticmethod
    def _get_ls_historic_metrics_updated(
        latest_metric: float, patience: int, ls_historic_metrics_current: List[float]
    ):
        ls_historic_metrics_updated = ls_historic_metrics_current.copy()
        ls_historic_metrics_updated.append(latest_metric)
        if len(ls_historic_metrics_updated) > 1 + patience:
            ls_historic_metrics_updated.pop(0)
        return ls_historic_metrics_updated

    @staticmethod
    def _deltas_exceeded_threshold(
        ls_deltas: List[float], delta_threshold: float
    ) -> List[bool]:
        return list(map(lambda delta: delta < -delta_threshold, ls_deltas))

    @staticmethod
    def _get_deltas(
        ls_successive_metrics: List[Tuple[float, float]],
    ) -> List[float]:
        return list(map(lambda tup_xy: tup_xy[1] - tup_xy[0], ls_successive_metrics))

    @staticmethod
    def _get_successive_metrics(
        ls_historic_metrics: List[float],
        patience: int,
    ) -> List[Tuple[float, float]]:
        ls_historic_metrics = ls_historic_metrics[-(1 + patience) : :]
        return [
            (ls_historic_metrics[i], ls_historic_metrics[i + 1])
            for i in range(len(ls_historic_metrics) - 1)
        ]
