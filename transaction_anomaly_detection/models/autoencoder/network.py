from typing import Optional, List, Tuple, Dict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transaction_anomaly_detection.models.layers.batch_swap_noise import (
    BatchSwapNoise,
)
from transaction_anomaly_detection.models.layers.continuous_feature_normalizer import (
    ContinuousFeatureNormalizer,
)
from transaction_anomaly_detection.models.layers.lin_bin_drop import LinBnDrop
from transaction_anomaly_detection.models.layers.reconstruction_loss import (
    ReconstructionLoss,
)
from transaction_anomaly_detection.models.layers.shifted_sigmoid import ShiftedSigmoid


class Autoencoder(nn.Module):
    @staticmethod
    def _get_dict_cat_feature_to_ls_categories(
        dict_cat_feature_to_ls_categories_n_embd: Dict[str, Tuple[List[str], int]]
    ) -> Dict[str, List[str]]:
        return {
            cat_feature: ls_categories_n_embd[0]
            for cat_feature, ls_categories_n_embd in dict_cat_feature_to_ls_categories_n_embd.items()
        }

    @staticmethod
    def _get_dict_cat_feature_to_n_embd(
        dict_cat_feature_to_ls_categories_n_embd: Dict[str, Tuple[List[str], int]]
    ) -> Dict[str, int]:
        return {
            cat_feature: ls_categories_n_embd[1]
            for cat_feature, ls_categories_n_embd in dict_cat_feature_to_ls_categories_n_embd.items()
        }

    @staticmethod
    def _get_ls_cat_features(
        dict_cat_feature_to_ls_categories_n_embd: Dict[str, Tuple[List[str], int]]
    ) -> List[str]:
        return list(dict_cat_feature_to_ls_categories_n_embd.keys())

    @staticmethod
    def _get_ls_n_categories(
        dict_cat_feature_to_ls_categories_n_embd: Dict[str, Tuple[List[str], int]]
    ) -> List[int]:
        return [
            len(ls_categories_n_embd[0])
            for ls_categories_n_embd in dict_cat_feature_to_ls_categories_n_embd.values()
        ]
