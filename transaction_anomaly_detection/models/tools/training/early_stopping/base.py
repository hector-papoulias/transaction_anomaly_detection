from typing import Tuple, Union, Optional
from abc import ABC, abstractmethod
import numpy as np
from torch.nn import Module


class EarlyStopperBase(ABC):
    def __init__(self, max_n_epochs: Optional[Union[int, float]] = np.inf):
        self._max_n_epochs = max_n_epochs
        self._n_epochs_ellapsed = 0
        self._stop: bool = False
        self._best_epoch: int = 0
        self._best_metric: float = np.inf
        self._best_model: Optional[Module] = None

    @property
    def max_n_epochs(self) -> Union[int, float]:
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
    def best_model(self) -> Module:
        return self._best_model

    def update(self, metric: float, model: Module):
        self._update_best_epoch_results(metric=metric, model=model)
        self.update_stop_state()
        self._n_epochs_ellapsed += 1

    @abstractmethod
    def update_stop_state(self):
        pass

    @classmethod
    @abstractmethod
    def stopping_condition_met(
        cls,
        n_epochs_ellapsed: int,
        max_n_epochs: Union[int, float],
    ) -> bool:
        return n_epochs_ellapsed >= max_n_epochs - 1

    def _update_best_epoch_results(self, metric: float, model: Module):
        (
            self._best_epoch,
            self._best_metric,
            self._best_model,
        ) = self._get_best_epoch_results(
            current_best_epoch=self._best_epoch,
            latest_epoch=self._n_epochs_ellapsed,
            current_best_metric=self._best_metric,
            latest_metric=metric,
            current_best_model=self._best_model,
            latest_model=model,
        )

    @classmethod
    def _get_best_epoch_results(
        cls,
        current_best_epoch: int,
        latest_epoch: int,
        current_best_metric: float,
        latest_metric: float,
        current_best_model: Module,
        latest_model: Module,
    ) -> Tuple[int, float, Module]:
        if cls._latest_metric_is_better_than_current_best(
            latest_metric=latest_metric, current_best_metric=current_best_metric
        ):
            return latest_epoch, latest_metric, latest_model
        else:
            return current_best_epoch, current_best_metric, current_best_model

    @staticmethod
    def _latest_metric_is_better_than_current_best(
        latest_metric: float, current_best_metric: float
    ) -> bool:
        return latest_metric <= current_best_metric
