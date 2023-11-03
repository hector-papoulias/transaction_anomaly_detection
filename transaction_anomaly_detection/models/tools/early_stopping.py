from typing import List, Tuple, Optional
import numpy as np
import torch.nn as nn


class EarlyStopper:
    def __init__(
        self, patience: int, delta_threshold: float, max_n_epochs: Optional[int] = None
    ):
        self._patience = patience
        self._delta_threshold = delta_threshold
        self._max_n_epochs = max_n_epochs if max_n_epochs is not None else np.inf
        self._n_epochs_ellapsed = 0
        self._stop: bool = False
        self._ls_historic_metrics: List[float] = []
        self._best_epoch: int = 0
        self._best_metric: float = np.inf
        self._best_model: Optional[nn.Module] = None

    @property
    def patience(self) -> int:
        return self._patience

    @property
    def delta_threshold(self) -> float:
        return self._delta_threshold

    @property
    def max_n_epochs(self) -> int:
        return self._max_n_epochs

    @property
    def n_epochs_ellapsed(self) -> int:
        return self._n_epochs_ellapsed

    @property
    def stop(self) -> bool:
        return self._stop

    @property
    def best_epoch(self) -> int:
        return self._best_epoch

    @property
    def best_metric(self) -> float:
        return self._best_metric

    @property
    def best_model(self) -> nn.Module:
        return self._best_model
    @staticmethod
    def _get_ls_historic_metrics_updated(
        latest_metric: float, patience: int, ls_historic_metrics_current: List[float]
    ):
        ls_historic_metrics_updated = ls_historic_metrics_current.copy()
        ls_historic_metrics_updated.append(latest_metric)
        if len(ls_historic_metrics_updated) > 1 + patience:
            ls_historic_metrics_updated.pop(0)
        return ls_historic_metrics_updated

    @classmethod
    def _stopping_condition_met(
        cls,
        patience: int,
        delta_threshold: float,
        max_n_epochs: int,
        n_epochs_ellapsed: int,
        ls_historic_metrics: List[float],
    ) -> bool:
        if n_epochs_ellapsed < 1 + patience:
            return False
        if n_epochs_ellapsed > max_n_epochs:
            return True
        ls_successive_metrics = cls._get_successive_metrics(
            ls_historic_metrics=ls_historic_metrics, patience=patience
        )
        ls_deltas = cls._get_deltas(ls_successive_metrics=ls_successive_metrics)
        ls_deltas_exceeded_threshold = cls._deltas_exceeded_threshold(
            ls_deltas=ls_deltas, delta_threshold=delta_threshold
        )
        return not any(ls_deltas_exceeded_threshold)

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
