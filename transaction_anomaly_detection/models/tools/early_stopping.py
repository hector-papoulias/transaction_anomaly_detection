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
