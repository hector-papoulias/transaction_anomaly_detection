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
    _dropout_rate_cat_decoder: float = 0.1

    def __init__(
        self,
        dict_cat_feature_to_ls_categories_n_embd: Dict[str, Tuple[List[str], int]],
        ls_con_features: List[int],
        ae_activation: nn.Module,
        encoder_layer_szs: List[int],
        dropout_rate: float,
        batchswap_noise_rate: float,
    ):
        super().__init__()
        # Extract and Save Categorical Feature Specs
        self._dict_cat_feature_to_ls_categories: Dict[
            str, List[str]
        ] = self._get_dict_cat_feature_to_ls_categories(
            dict_cat_feature_to_ls_categories_n_embd=dict_cat_feature_to_ls_categories_n_embd
        )
        self._dict_cat_feature_to_n_embd: Dict[
            str, int
        ] = self._get_dict_cat_feature_to_n_embd(
            dict_cat_feature_to_ls_categories_n_embd=dict_cat_feature_to_ls_categories_n_embd
        )
        self._ls_cat_features: List[str] = self._get_ls_cat_features(
            dict_cat_feature_to_ls_categories_n_embd=dict_cat_feature_to_ls_categories_n_embd
        )
        self._ls_n_categories: List[int] = self._get_ls_n_categories(
            dict_cat_feature_to_ls_categories_n_embd=dict_cat_feature_to_ls_categories_n_embd
        )
        self._n_categorical_features: int = len(self._ls_cat_features)
        n_categories_total = sum(self._ls_n_categories)
        ls_n_embd = list(self._dict_cat_feature_to_n_embd.values())
        n_embed_total = sum(ls_n_embd)
        self._has_cat: bool = True if self._n_categorical_features > 0 else False
        # Extract and Save Continuous Feature Specs
        self._ls_con_features = ls_con_features
        self._n_continuous_features = len(self._ls_con_features)
        self._has_con = True if self._n_continuous_features > 0 else False
        if self._has_con:
            self._df_con_stats = self._get_default_df_con_stats(
                ls_con_features=self._ls_con_features
            )
        else:
            self._df_con_stats = None
        # Compute Standard Autoencoder External Dimension
        dim_ae_external = self._n_continuous_features + n_embed_total
        # Save Loss
        self._loss = ReconstructionLoss()
        # Save Batch Swap Noise Preprocessor
        self._batchswap_noise = BatchSwapNoise(swap_rate=batchswap_noise_rate)
        # Save Categorical Feature Preprocessor
        if self._has_cat:
            self._cat_preprocessor = self._get_cat_preprocessor(
                dict_cat_feature_to_ls_categories_n_embd=dict_cat_feature_to_ls_categories_n_embd,
                dropout_rate=dropout_rate,
            )
        # Save Continuous Feature Preprocessor
        if self._has_con:
            self._con_preprocessor = self._get_con_preprocessor(
                n_continuous_features=self._n_continuous_features,
                sr_feature_means=self._df_con_stats["mean"],
                sr_feature_stds=self._df_con_stats["std"],
            )
        # Save Standard AutoEncoder
        self._standard_autoencoder = self._get_standard_autoencoder(
            dim_external=dim_ae_external,
            encoder_layer_szs=encoder_layer_szs,
            activation=ae_activation,
            bn=True,
            dropout_rate=dropout_rate,
        )
        # Save Continuous Feature Postprocessor
        if self._has_con:
            self._con_postprocessor = self._get_con_postprocessor(
                dim_autoencoder_output=dim_ae_external,
                n_continuous_features=self._n_continuous_features,
                sr_min=self._df_con_stats["min"],
                sr_max=self._df_con_stats["max"],
                sr_mean=self._df_con_stats["mean"],
                sr_std=self._df_con_stats["std"],
            )
        # Save Categorical Feature Postprocessor
        if self._has_cat:
            self._cat_postprocessor = self._get_cat_postprocessor(
                dim_autoencoder_output=dim_ae_external,
                n_categories_total=n_categories_total,
                dropout_rate=self._dropout_rate_cat_decoder,
            )

    @property
    def has_cat(self) -> bool:
        return self._has_cat

    @property
    def ls_cat_features(self) -> List[str]:
        return self._ls_cat_features

    @property
    def dict_cat_feature_to_ls_categories(self) -> Dict[str, List[str]]:
        return self._dict_cat_feature_to_ls_categories

    @property
    def dict_cat_feature_to_n_embd(self) -> Dict[str, int]:
        return self._dict_cat_feature_to_n_embd

    @property
    def has_con(self) -> bool:
        return self._has_con

    @property
    def ls_con_features(self) -> List[str]:
        return self._ls_con_features

    @property
    def df_con_stats(self) -> Optional[pd.DataFrame]:
        return self._df_con_stats

    def fit_continuous_feature_statistics(
        self,
        df_dataset: pd.DataFrame,
    ):
        if self._has_con:
            self._df_con_stats = self._get_fitted_df_con_stats(
                df_dataset=df_dataset, ls_con_features=self._ls_con_features
            )
            self._con_preprocessor["normalizer"].means = torch.tensor(
                list(self._df_con_stats["mean"].values)
            )
            self._con_preprocessor["normalizer"].stds = torch.tensor(
                list(self._df_con_stats["std"].values)
            )
            sr_low = self._get_sr_low(
                sr_min=self._df_con_stats["min"],
                sr_mean=self._df_con_stats["mean"],
                sr_std=self._df_con_stats["std"],
            )
            sr_high = self._get_sr_high(
                sr_max=self._df_con_stats["max"],
                sr_mean=self._df_con_stats["mean"],
                sr_std=self._df_con_stats["std"],
            )
            shifted_sigmoid = self._con_postprocessor["t_decoded_to_zscores"][-1]
            shifted_sigmoid.min_vals = torch.tensor(list(sr_low.values))
            shifted_sigmoid.max_vals = torch.tensor(list(sr_high.values))

    def forward(
        self,
        t_in: torch.tensor,
        argmax_cat_logits: Optional[bool] = False,
        denormalize_con_outputs: Optional[bool] = False,
        compute_loss: Optional[bool] = False,
        loss_batch_reduction: Optional[str] = None,
    ) -> Tuple[
        torch.tensor,
        Tuple[torch.tensor],
        torch.tensor,
        torch.tensor,
        torch.tensor,
        torch.tensor,
    ]:
        # BatchSwapNoise Augmentation
        t_in = self._batchswap_noise(t_in)
        # t_in shape: (B, n_categorical_features + n_continuous_features)
        # BatchSwapNoise doesn't change the shape.

        # Preprocessing
        t_cat_embd, tup_t_cat_targets = None, None
        if self._has_cat:
            t_in_cat = self._get_t_in_cat(
                t_in=t_in, n_categorical_features=self._n_categorical_features
            )  # t_in_cat shape: (B, n_categorical_features)
            t_cat_embd, tup_t_cat_targets = self._preprocess_cat_features(
                t_in_cat=t_in_cat, cat_preprocessor=self._cat_preprocessor
            )
            # t_cat_embd shape: (B, sum(n_embd)),
            # tup_t_cat_targets contains n_categorical_features tensors, each of shape: (B)
        t_in_con_zscores, t_con_zscore_targets = None, None
        if self._has_con:
            t_in_con = self._get_t_in_con(
                t_in=t_in, n_continuous_features=self._n_continuous_features
            )  # t_in_con shape: (B, n_continuous_features)
            t_in_con_zscores, t_con_zscore_targets = self._preprocess_con_features(
                t_in_con=t_in_con, con_preprocessor=self._con_preprocessor
            )
            # t_in_con_zscores shape: (B, n_continuous_features),
            # t_con_targets_shape (B, n_continuous_features)

        # Standard Autoencoder Forward Pass
        t_encoded, t_decoded = self._pass_through_standard_autoencoder(
            t_cat_embd=t_cat_embd,
            t_in_con_zscores=t_in_con_zscores,
            standard_autoencoder=self._standard_autoencoder,
        )
        # t_encoded shape: (B, encoder_layer_szs[-1])
        # t_decoded shape: (B, dim_ae_external) = (B, n_con_features+sum(n_embd_cat))

        # Reconstructions
        tup_t_cat_logits, tup_t_cat_reconstructions = None, None
        if self._has_cat:
            (
                tup_t_cat_logits,
                tup_t_cat_reconstructions,
            ) = self._reconstruct_cat_features(
                t_decoded=t_decoded,
                ls_n_categories=self._ls_n_categories,
                cat_postprocessor=self._cat_postprocessor,
                argmax_cat_logits=argmax_cat_logits,
            )
        # tup_t_cat_logits: tuple containing a tensor of shape (B, n_categories) for each categorical feature.
        # tup_t_cat_reconstructions: tuple containing a tensor of shape (B) for each categorical feature.
        t_out_con_zscores, t_out_con_reconstructions = None, None
        if self._has_con:
            (
                t_out_con_zscores,
                t_out_con_reconstructions,
            ) = self._reconstruct_con_features(
                t_decoded=t_decoded,
                con_preprocessor=self._con_preprocessor,
                con_postprocessor=self._con_postprocessor,
                denormalize_con_outputs=denormalize_con_outputs,
            )
        # t_out_con_zscores shape: (B, n_con_features)
        # t_out_con_reconstructions shape: (B, n_con_features)

        # Reconstruction Output Selection
        tup_t_out_cat = self._get_cat_output(
            tup_t_cat_logits=tup_t_cat_logits,
            tup_t_cat_reconstructions=tup_t_cat_reconstructions,
        )
        # tup_t_out_cat: tuple containing a tensor for each categorical feature.
        # The shapes are either all (B, n_categories)---logits--- or (B)---reconstructions
        t_out_con = self._get_con_output(
            t_out_con_zscores=t_out_con_zscores,
            t_out_con_reconstructions=t_out_con_reconstructions,
        )
        # t_out_con shape: (B, n_con_features)

        # Reconstruction Loss Computation
        t_loss, t_cat_losses, t_con_losses = None, None, None
        if compute_loss:
            t_loss, t_cat_losses, t_con_losses = self._compute_reconstruction_loss(
                loss_module=self._loss,
                tup_t_cat_logits=tup_t_cat_logits,
                tup_t_cat_targets=tup_t_cat_targets,
                t_out_con_zscores=t_out_con_zscores,
                t_con_zscore_targets=t_con_zscore_targets,
                loss_batch_reduction=loss_batch_reduction,
            )
            # t_cat_losses shape: (B, n_cat_features) if no batch_reduction is applied, (n_cat_features) otherwise.
            # t_con_losses shape: (B, n_con_features) if no batch_reduction is applied, (n_con_features) otherwise.
            # t_loss shape: (B) if no batch_reduction is applied, () otherwise.
        return t_encoded, tup_t_out_cat, t_out_con, t_loss, t_cat_losses, t_con_losses

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

    @staticmethod
    def _get_con_preprocessor(
        n_continuous_features: int,
        sr_feature_means: pd.Series,
        sr_feature_stds: pd.Series,
    ) -> nn.ModuleDict:
        con_preprocessor = nn.ModuleDict()
        con_preprocessor["normalizer"] = ContinuousFeatureNormalizer(
            t_means=torch.tensor(
                sr_feature_means.tolist(), dtype=torch.float, requires_grad=False
            ),
            t_stds=torch.tensor(
                sr_feature_stds.tolist(), dtype=torch.float, requires_grad=False
            ),
        )
        con_preprocessor["batch_norm"] = nn.BatchNorm1d(n_continuous_features)
        return con_preprocessor

    @staticmethod
    def _get_cat_preprocessor(
        dict_cat_feature_to_ls_categories_n_embd: Dict[str, Tuple[List[str], int]],
        dropout_rate: float,
    ) -> nn.ModuleDict:
        cat_preprocessor = nn.ModuleDict()
        cat_preprocessor["embeddings"] = nn.ModuleList(
            [
                nn.Embedding(
                    num_embeddings=len(ls_categories), embedding_dim=embedding_dim
                )
                for ls_categories, embedding_dim in dict_cat_feature_to_ls_categories_n_embd.values()
            ]
        )
        cat_preprocessor["dropout"] = nn.Dropout(dropout_rate)
        return cat_preprocessor

    @classmethod
    def _get_standard_autoencoder(
        cls,
        dim_external: int,
        encoder_layer_szs: List[int],
        activation: nn.Module,
        bn: bool,
        dropout_rate: float,
    ) -> nn.ModuleDict:
        standard_autoencoder = nn.ModuleDict()
        ls_encoder_layer_szs = [dim_external] + encoder_layer_szs
        standard_autoencoder["encoder"] = cls._get_mlp(
            layer_szs=ls_encoder_layer_szs,
            activation=activation,
            bn=bn,
            dropout_rate=dropout_rate,
        )
        standard_autoencoder["decoder"] = cls._get_mlp(
            layer_szs=ls_encoder_layer_szs[::-1],
            activation=activation,
            bn=bn,
            dropout_rate=dropout_rate,
        )
        return standard_autoencoder

    @staticmethod
    def _get_mlp(
        layer_szs: List[int], activation: nn.Module, bn: bool, dropout_rate: float
    ) -> nn.Sequential:
        mlp = nn.Sequential(
            *[
                LinBnDrop(
                    dim_in=dim_in,
                    dim_out=dim_out,
                    activation=activation,
                    bn=bn,
                    dropout_rate=dropout_rate,
                )
                for dim_in, dim_out in zip(layer_szs, layer_szs[1:])
            ]
        )
        return mlp

    @staticmethod
    def _get_cat_postprocessor(
        dim_autoencoder_output: int, n_categories_total: int, dropout_rate: float
    ) -> nn.ModuleDict:
        cat_postprocessor = nn.ModuleDict()
        cat_postprocessor["t_decoded_to_logits"] = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(
                in_features=dim_autoencoder_output,
                out_features=n_categories_total,
                bias=True,
            ),
        )
        return cat_postprocessor

    @classmethod
    def _get_con_postprocessor(
        cls,
        dim_autoencoder_output: int,
        n_continuous_features: int,
        sr_min: pd.Series,
        sr_max: pd.Series,
        sr_mean: pd.Series,
        sr_std: pd.Series,
    ) -> nn.ModuleDict:
        con_postprocessor = nn.ModuleDict()
        sr_low = cls._get_sr_low(sr_min=sr_min, sr_mean=sr_mean, sr_std=sr_std)
        sr_high = cls._get_sr_high(sr_max=sr_max, sr_mean=sr_mean, sr_std=sr_std)
        con_postprocessor["t_decoded_to_zscores"] = nn.Sequential(
            LinBnDrop(
                dim_in=dim_autoencoder_output,
                dim_out=n_continuous_features,
                activation=None,
                bn=False,
                dropout_rate=0,
            ),
            ShiftedSigmoid(
                t_min_values=torch.tensor(list(sr_low.values), requires_grad=False),
                t_max_values=torch.tensor(list(sr_high.values), requires_grad=False),
            ),
        )
        return con_postprocessor

    @staticmethod
    def _get_t_in_cat(t_in: torch.tensor, n_categorical_features: int) -> torch.tensor:
        return t_in[:, :n_categorical_features].to(torch.int)

    @staticmethod
    def _get_t_in_con(t_in: torch.tensor, n_continuous_features) -> torch.tensor:
        return t_in[:, -n_continuous_features:].to(torch.float)

    @staticmethod
    def _preprocess_cat_features(
        t_in_cat: torch.tensor, cat_preprocessor: nn.ModuleDict
    ) -> Tuple[torch.tensor, Tuple[torch.tensor]]:
        t_cat_targets = t_in_cat
        tup_t_cat_targets = torch.unbind(input=t_cat_targets, dim=-1)
        ls_t_cat_embd = [
            embed(t_in_cat[:, i].long())
            for i, embed in enumerate(cat_preprocessor["embeddings"])
        ]
        ls_t_cat_embd = list(
            map(
                lambda t_cat_embd: cat_preprocessor["dropout"](t_cat_embd),
                ls_t_cat_embd,
            )
        )
        t_cat_embd = torch.cat(tensors=ls_t_cat_embd, dim=-1)
        return t_cat_embd, tup_t_cat_targets

    @staticmethod
    def _preprocess_con_features(
        t_in_con: torch.tensor, con_preprocessor: nn.ModuleDict
    ) -> Tuple[torch.tensor, torch.tensor]:
        t_in_con_zscores = con_preprocessor["normalizer"](t_in_con)
        t_con_zscore_targets = t_in_con_zscores
        t_in_con_zscores = con_preprocessor["batch_norm"](t_in_con)
        return t_in_con_zscores, t_con_zscore_targets

    @staticmethod
    def _pass_through_standard_autoencoder(
        t_cat_embd: torch.tensor,
        t_in_con_zscores: torch.tensor,
        standard_autoencoder: nn.ModuleDict,
    ) -> torch.tensor:
        ae_input = torch.cat(
            tensors=[t for t in [t_cat_embd, t_in_con_zscores] if t is not None], dim=-1
        )
        t_encoded = standard_autoencoder["encoder"](ae_input)
        t_decoded = standard_autoencoder["decoder"](t_encoded)
        return t_encoded, t_decoded

    @staticmethod
    def _reconstruct_cat_features(
        t_decoded: torch.tensor,
        ls_n_categories: List[int],
        cat_postprocessor: nn.ModuleDict,
        argmax_cat_logits: bool,
    ) -> Tuple[Tuple[torch.tensor], Optional[Tuple[torch.tensor]]]:
        t_out_cat_tot = cat_postprocessor["t_decoded_to_logits"](t_decoded)
        tup_t_cat_logits = torch.split(
            tensor=t_out_cat_tot,
            split_size_or_sections=ls_n_categories,
            dim=-1,
        )
        tup_t_cat_reconstructions = None
        if argmax_cat_logits:
            ls_t_cat_reconstructions = []
            for t_cat_logits in tup_t_cat_logits:
                ls_t_cat_reconstructions.append(torch.argmax(t_cat_logits, dim=-1))
            tup_t_cat_reconstructions = tuple(ls_t_cat_reconstructions)
        return tup_t_cat_logits, tup_t_cat_reconstructions

    @staticmethod
    def _get_cat_output(
        tup_t_cat_logits: Tuple[torch.tensor],
        tup_t_cat_reconstructions: Optional[Tuple[torch.tensor]],
    ) -> Tuple[torch.tensor]:
        tup_t_out_cat = (
            tup_t_cat_reconstructions
            if tup_t_cat_reconstructions is not None
            else tup_t_cat_logits
        )
        return tup_t_out_cat

    @staticmethod
    def _reconstruct_con_features(
        t_decoded: torch.tensor,
        con_preprocessor: nn.ModuleDict,
        con_postprocessor: nn.ModuleDict,
        denormalize_con_outputs: bool,
    ) -> Tuple[torch.tensor, Optional[torch.tensor]]:
        t_out_con_zscores = con_postprocessor["t_decoded_to_zscores"](t_decoded)
        t_out_con_reconstructions = None
        if denormalize_con_outputs:
            t_out_con_reconstructions = con_preprocessor["normalizer"](
                t_out_con_zscores, denormalize=True
            )
        return t_out_con_zscores, t_out_con_reconstructions

    @staticmethod
    def _get_con_output(
        t_out_con_zscores: torch.tensor,
        t_out_con_reconstructions: Optional[torch.tensor],
    ) -> torch.tensor:
        t_out_con = (
            t_out_con_reconstructions
            if t_out_con_reconstructions is not None
            else t_out_con_zscores
        )
        return t_out_con

    @staticmethod
    def _compute_reconstruction_loss(
        loss_module: ReconstructionLoss,
        tup_t_cat_logits: Optional[Tuple[torch.tensor]] = None,
        tup_t_cat_targets: Optional[Tuple[torch.tensor]] = None,
        t_out_con_zscores: Optional[torch.tensor] = None,
        t_con_zscore_targets: Optional[torch.tensor] = None,
        loss_batch_reduction: Optional[str] = None,
    ) -> Tuple[torch.tensor, Optional[torch.tensor], Optional[torch.tensor]]:
        t_cat_losses, t_con_losses = loss_module(
            ls_cat_logits=tup_t_cat_logits,
            ls_cat_targets=tup_t_cat_targets,
            t_con_predictions=t_out_con_zscores,
            t_con_targets=t_con_zscore_targets,
            batch_reduction=loss_batch_reduction,
        )
        t_loss = torch.mean(
            torch.cat(
                [t for t in [t_cat_losses, t_con_losses] if t is not None], axis=-1
            ),  # Shape (B, n_cat_features + n_con_features) if no reduction is applied, (n_cat_features + n_con_features) otherwise.
            axis=-1,
        )
        return t_loss, t_cat_losses, t_con_losses

    @staticmethod
    def _get_sr_low(
        sr_min: pd.Series, sr_mean: pd.Series, sr_std: pd.Series
    ) -> pd.Series:
        return (sr_min - sr_mean) / sr_std

    @staticmethod
    def _get_sr_high(
        sr_max: pd.Series, sr_mean: pd.Series, sr_std: pd.Series
    ) -> pd.Series:
        return (sr_max - sr_mean) / sr_std

    @classmethod
    def _get_fitted_df_con_stats(
        cls,
        df_dataset: pd.DataFrame,
        ls_con_features: List[str],
    ) -> pd.DataFrame:
        dict_con_feature_to_mean_std_min_max = {}
        for con_feature in ls_con_features:
            dict_con_feature_to_mean_std_min_max[
                con_feature
            ] = cls._get_mean_std_min_max(arr_vals=df_dataset[con_feature])
        df_con_stats = cls._dict_con_stats_to_df_con_stats(
            dict_con_feature_to_mean_std_min_max=dict_con_feature_to_mean_std_min_max
        )
        return df_con_stats

    @classmethod
    def _get_default_df_con_stats(cls, ls_con_features: List[str]) -> pd.DataFrame:
        dict_con_feature_to_mean_std_min_max = {}
        for con_feature in ls_con_features:
            dict_con_feature_to_mean_std_min_max[con_feature] = (0, 1, -1, 1)
        df_con_stats = cls._dict_con_stats_to_df_con_stats(
            dict_con_feature_to_mean_std_min_max=dict_con_feature_to_mean_std_min_max
        )
        return df_con_stats

    @staticmethod
    def _dict_con_stats_to_df_con_stats(
        dict_con_feature_to_mean_std_min_max: Dict[
            str, Tuple[float, float, float, float]
        ]
    ) -> pd.DataFrame:
        df_con_stats = pd.DataFrame.from_dict(
            data=dict_con_feature_to_mean_std_min_max,
            orient="index",
            columns=["mean", "std", "min", "max"],
        )
        return df_con_stats

    @staticmethod
    def _get_mean_std_min_max(arr_vals: np.array) -> Tuple[float, float, float, float]:
        mean = np.nanmean(arr_vals)
        std = np.nanstd(arr_vals)
        if std == 0:
            std = 1
        minimum = np.nanmin(arr_vals)
        maximum = np.nanmax(arr_vals)
        return mean, std, minimum, maximum
