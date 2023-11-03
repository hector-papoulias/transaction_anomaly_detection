from typing import List, Dict, Optional, Union, Tuple, Generator, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transaction_anomaly_detection.models.tools.tokenizer import Tokenizer
from transaction_anomaly_detection.models.autoencoder.network import Autoencoder
from transaction_anomaly_detection.models.autoencoder.trainer import AutoencoderTrainer


class TransactionAnomalyDetector:
    def __init__(
        self,
        dict_cat_feature_to_ls_categories_n_embd: Dict[str, Tuple[List[str], int]],
        ls_con_features: List[str],
        encoder_layer_szs: List[int],
        ae_activation: nn.Module,
        dropout_rate: Optional[float] = 0,
        batchswap_noise_rate: Optional[float] = 0,
    ):
        self._autoencoder = Autoencoder(
            dict_cat_feature_to_ls_categories_n_embd=dict_cat_feature_to_ls_categories_n_embd,
            ls_con_features=ls_con_features,
            encoder_layer_szs=encoder_layer_szs,
            ae_activation=ae_activation,
            dropout_rate=dropout_rate,
            batchswap_noise_rate=batchswap_noise_rate,
        )
        self._autoencoder.eval()
        self._dict_cat_feature_to_tokenizer = self._get_dict_cat_feature_to_tokenizer(
            dict_cat_feature_to_ls_categories=self._autoencoder.dict_cat_feature_to_ls_categories
        )
        self._reconstruction_loss_threshold: float = np.nan

    @property
    def has_cat(self) -> bool:
        return self._autoencoder.has_cat

    @property
    def ls_cat_features(self) -> List[str]:
        return self._autoencoder.ls_cat_features

    @property
    def dict_cat_feature_to_ls_categories(self) -> Dict[str, List[str]]:
        return self._autoencoder.dict_cat_feature_to_ls_categories

    @property
    def dict_cat_feature_to_n_embd(self) -> Dict[str, int]:
        return self._autoencoder.dict_cat_feature_to_n_embd

    @property
    def has_con(self) -> bool:
        return self._autoencoder.has_con

    @property
    def ls_con_features(self) -> List[str]:
        return self._autoencoder.ls_con_features

    @property
    def df_con_stats(self) -> Optional[pd.DataFrame]:
        return self._autoencoder.df_con_stats

    @property
    def autoencoder_architecture(self) -> str:
        return repr(self._autoencoder)

    @property
    def reconstruction_loss_threshold(self) -> float:
        return self._reconstruction_loss_threshold

    def fit(
        self,
        df_dataset: pd.DataFrame,
        contamination: float,
        val_ratio: float,
        sz_batch: int,
        learning_rate: float,
        patience: int,
        loss_delta_threshold: float,
        max_n_epochs: Optional[int] = None,
        verbose: Optional[bool] = False,
    ) -> Dict[str, pd.Series]:
        self._autoencoder.fit_continuous_feature_statistics(df_dataset=df_dataset)
        t_dataset = self._prepare_t_input(
            input_data=df_dataset,
            ls_cat_features=self._autoencoder.ls_cat_features,
            ls_con_features=self._autoencoder.ls_con_features,
            dict_cat_feature_to_tokenizer=self._dict_cat_feature_to_tokenizer,
        )
        self._autoencoder, dict_loss_evolution = AutoencoderTrainer.train(
            autoencoder=self._autoencoder,
            t_dataset=t_dataset,
            val_ratio=val_ratio,
            sz_batch=sz_batch,
            learning_rate=learning_rate,
            patience=patience,
            loss_delta_threshold=loss_delta_threshold,
            max_n_epochs=max_n_epochs,
            verbose=verbose,
        )
        self.fit_reconstruction_loss_threshold(
            df_dataset=df_dataset, contamination=contamination
        )
        return dict_loss_evolution

    def fit_reconstruction_loss_threshold(
        self, df_dataset: pd.DataFrame, contamination: float
    ):
        threshold = self._compute_reconstruction_loss_threshold(
            quantile=1 - contamination,
            autoencoder=self._autoencoder,
            input_data=df_dataset,
            ls_cat_features=self._autoencoder.ls_cat_features,
            ls_con_features=self._autoencoder.ls_con_features,
            dict_cat_feature_to_tokenizer=self._dict_cat_feature_to_tokenizer,
        )
        self._reconstruction_loss_threshold = threshold

    def detect_anomalies(
        self, input_data: Union[pd.Series, pd.DataFrame]
    ) -> pd.DataFrame:
        return self._detect_anomalies(
            reconstruction_loss_threshold=self._reconstruction_loss_threshold,
            autoencoder=self._autoencoder,
            input_data=input_data,
            ls_cat_features=self._autoencoder.ls_cat_features,
            ls_con_features=self._autoencoder.ls_con_features,
            dict_cat_feature_to_tokenizer=self._dict_cat_feature_to_tokenizer,
        )

    def reconstruct(
        self,
        input_data: Union[pd.Series, pd.DataFrame],
    ) -> pd.DataFrame:
        return self._reconstruct(
            autoencoder=self._autoencoder,
            input_data=input_data,
            ls_cat_features=self._autoencoder.ls_cat_features,
            ls_con_features=self._autoencoder.ls_con_features,
            dict_cat_feature_to_tokenizer=self._dict_cat_feature_to_tokenizer,
        )

    def encode(
        self,
        input_data: Union[pd.Series, pd.DataFrame],
    ) -> torch.tensor:
        return self._encode(
            autoencoder=self._autoencoder,
            input_data=input_data,
            ls_cat_features=self._autoencoder.ls_cat_features,
            ls_con_features=self._autoencoder.ls_con_features,
            dict_cat_feature_to_tokenizer=self._dict_cat_feature_to_tokenizer,
        )

    def compute_mean_reconstruction_loss(
        self,
        input_data: Union[pd.Series, pd.DataFrame],
    ) -> float:
        return self._compute_mean_reconstruction_loss(
            autoencoder=self._autoencoder,
            input_data=input_data,
            ls_cat_features=self._autoencoder.ls_cat_features,
            ls_con_features=self._autoencoder.ls_con_features,
            dict_cat_feature_to_tokenizercat_feature_to_tokenizer=self._dict_cat_feature_to_tokenizer,
        )

    def compute_reconstruction_loss_by_record(
        self,
        input_data: Union[pd.Series, pd.DataFrame],
    ) -> pd.Series:
        return self._compute_reconstruction_loss_by_record(
            autoencoder=self._autoencoder,
            input_data=input_data,
            ls_cat_features=self._autoencoder.ls_cat_features,
            ls_con_features=self._autoencoder.ls_con_features,
            dict_cat_feature_to_tokenizer=self._dict_cat_feature_to_tokenizer,
        )

    def compute_reconstruction_loss_by_feature(
        self,
        input_data: Union[pd.Series, pd.DataFrame],
    ) -> pd.Series:
        return self._compute_reconstruction_loss_by_feature(
            autoencoder=self._autoencoder,
            input_data=input_data,
            ls_cat_features=self._autoencoder.ls_cat_features,
            ls_con_features=self._autoencoder.ls_con_features,
            dict_cat_feature_to_tokenizer=self._dict_cat_feature_to_tokenizer,
        )

    @classmethod
    def _detect_anomalies(
        cls,
        autoencoder: Autoencoder,
        reconstruction_loss_threshold: float,
        input_data: Union[pd.Series, pd.DataFrame],
        ls_cat_features: List[str],
        ls_con_features: List[str],
        dict_cat_feature_to_tokenizer: Dict[str, Tokenizer],
    ) -> pd.DataFrame:
        sr_loss_by_record = cls._compute_reconstruction_loss_by_record(
            autoencoder=autoencoder,
            input_data=input_data,
            ls_cat_features=ls_cat_features,
            ls_con_features=ls_con_features,
            dict_cat_feature_to_tokenizer=dict_cat_feature_to_tokenizer,
        )
        idx_anomalies = cls._get_idx_anomalies(
            sr_loss_by_record=sr_loss_by_record,
            reconstruction_loss_threshold=reconstruction_loss_threshold,
        )
        df_anomalies = cls._format_df_anomalies(
            input_data=input_data,
            idx_anomalies=idx_anomalies,
            sr_loss_by_record=sr_loss_by_record,
        )
        return df_anomalies

    @classmethod
    def _reconstruct(
        cls,
        autoencoder: Autoencoder,
        input_data: Union[pd.Series, pd.DataFrame],
        ls_cat_features: List[str],
        ls_con_features: List[str],
        dict_cat_feature_to_tokenizer: Dict[str, Tokenizer],
    ) -> pd.DataFrame:
        ls_input_features = cls._extract_feature_names(input_data=input_data)
        t_input_data = cls._prepare_t_input(
            input_data=input_data,
            ls_cat_features=ls_cat_features,
            ls_con_features=ls_con_features,
            dict_cat_feature_to_tokenizer=dict_cat_feature_to_tokenizer,
        )
        (
            tup_t_cat_reconstructions,
            t_out_con_reconstructions,
        ) = cls._get_reconstruction_tensors(
            autoencoder=autoencoder,
            t_input_data=t_input_data,
            argmax_cat_logits=True,
            denormalize_con_outputs=True,
        )
        dict_reconstructions = cls._format_dict_reconstructions(
            ls_cat_features=ls_cat_features,
            ls_con_features=ls_con_features,
            dict_cat_feature_to_tokenizer=dict_cat_feature_to_tokenizer,
            tup_t_cat_reconstructions=tup_t_cat_reconstructions,
            t_out_con_reconstructions=t_out_con_reconstructions,
        )
        df_reconstructions = cls._format_df_reconstructions(
            ls_cols_original=ls_input_features,
            dict_reconstructions=dict_reconstructions,
        )
        return df_reconstructions

    @classmethod
    def _encode(
        cls,
        autoencoder: Autoencoder,
        input_data: Union[pd.Series, pd.DataFrame],
        ls_cat_features: List[str],
        ls_con_features: List[str],
        dict_cat_feature_to_tokenizer: Dict[str, Tokenizer],
    ) -> torch.tensor:
        t_input_data = cls._prepare_t_input(
            input_data=input_data,
            ls_cat_features=ls_cat_features,
            ls_con_features=ls_con_features,
            dict_cat_feature_to_tokenizer=dict_cat_feature_to_tokenizer,
        )
        t_latent_rep = cls._get_latent_rep_tensor(
            autoencoder=autoencoder, t_input_data=t_input_data
        )
        return t_latent_rep

    @classmethod
    def _compute_reconstruction_loss_threshold(
        cls,
        quantile: float,
        autoencoder: Autoencoder,
        input_data: pd.DataFrame,
        ls_cat_features: List[str],
        ls_con_features: List[str],
        dict_cat_feature_to_tokenizer: Dict[str, Tokenizer],
    ) -> float:
        sr_loss_by_record = cls._compute_reconstruction_loss_by_record(
            autoencoder=autoencoder,
            input_data=input_data,
            ls_cat_features=ls_cat_features,
            ls_con_features=ls_con_features,
            dict_cat_feature_to_tokenizer=dict_cat_feature_to_tokenizer,
        )
        return sr_loss_by_record.quantile(quantile)

    @classmethod
    def _compute_mean_reconstruction_loss(
        cls,
        autoencoder: Autoencoder,
        input_data: pd.DataFrame,
        ls_cat_features: List[str],
        ls_con_features: List[str],
        dict_cat_feature_to_tokenizer: Dict[str, Tokenizer],
    ) -> float:
        t_input_data = cls._prepare_t_input(
            input_data=input_data,
            ls_cat_features=ls_cat_features,
            ls_con_features=ls_con_features,
            dict_cat_feature_to_tokenizer=dict_cat_feature_to_tokenizer,
        )
        t_reconstruction_loss, _, _ = cls._get_loss_tensors(
            autoencoder=autoencoder,
            t_input_data=t_input_data,
            loss_batch_reduction="mean",
        )
        # t_reconstruction_loss shape: ()
        return t_reconstruction_loss.item()

    @classmethod
    def _compute_reconstruction_loss_by_record(
        cls,
        autoencoder: Autoencoder,
        input_data: pd.DataFrame,
        ls_cat_features: List[str],
        ls_con_features: List[str],
        dict_cat_feature_to_tokenizer: Dict[str, Tokenizer],
    ) -> pd.Series:
        t_input_data = cls._prepare_t_input(
            input_data=input_data,
            ls_cat_features=ls_cat_features,
            ls_con_features=ls_con_features,
            dict_cat_feature_to_tokenizer=dict_cat_feature_to_tokenizer,
        )
        t_reconstruction_loss, _, _ = cls._get_loss_tensors(
            autoencoder=autoencoder,
            t_input_data=t_input_data,
            loss_batch_reduction="none",
        )
        # t_reconstruction_loss shape: (B)
        sr_loss_by_record = cls._format_sr_loss_by_record(
            index=list(input_data.index), t_reconstruction_loss=t_reconstruction_loss
        )
        return sr_loss_by_record

    @classmethod
    def _compute_reconstruction_loss_by_feature(
        cls,
        autoencoder: Autoencoder,
        input_data: Union[pd.Series, pd.DataFrame],
        ls_cat_features: List[str],
        ls_con_features: List[str],
        dict_cat_feature_to_tokenizer: Dict[str, Tokenizer],
    ) -> pd.Series:
        t_input_data = cls._prepare_t_input(
            input_data=input_data,
            ls_cat_features=ls_cat_features,
            ls_con_features=ls_con_features,
            dict_cat_feature_to_tokenizer=dict_cat_feature_to_tokenizer,
        )
        _, t_cat_losses, t_con_losses = cls._get_loss_tensors(
            autoencoder=autoencoder,
            t_input_data=t_input_data,
            loss_batch_reduction="mean",
        )
        dict_loss_by_feature = cls._get_dict_loss_by_feature(
            ls_features=ls_cat_features + ls_con_features,
            t_losses=torch.cat([t_cat_losses, t_con_losses]),
        )
        sr_loss_by_feature = cls._format_sr_loss_by_feature(
            dict_loss_by_feature=dict_loss_by_feature
        )
        return sr_loss_by_feature

    @staticmethod
    def _format_df_anomalies(
        input_data: pd.DataFrame, idx_anomalies: List[int], sr_loss_by_record
    ) -> pd.DataFrame:
        df_anomalies = input_data.iloc[idx_anomalies].copy()
        if not df_anomalies.empty:
            df_anomalies["loss"] = sr_loss_by_record
            df_anomalies.sort_values(by="loss", ascending=False, inplace=True)
            df_anomalies.reset_index(inplace=True)
            df_anomalies.rename(columns={"index": "index"}, inplace=True)
            df_anomalies.rename_axis(index="loss_rank", inplace=True)
            df_anomalies = df_anomalies.loc[
                :, ["loss"] + [col for col in df_anomalies.columns if col != "loss"]
            ]
        return df_anomalies

    @staticmethod
    def _get_idx_anomalies(
        sr_loss_by_record: pd.Series, reconstruction_loss_threshold: float
    ) -> List[int]:
        return list(
            sr_loss_by_record[sr_loss_by_record > reconstruction_loss_threshold].index
        )

    @staticmethod
    def _format_df_reconstructions(
        ls_cols_original: List[str],
        dict_reconstructions: Dict[str, Union[List[str], List[float]]],
    ) -> pd.DataFrame:
        df_reconstructions = pd.DataFrame(dict_reconstructions)
        df_reconstructions = df_reconstructions.loc[
            :,
            [
                feature
                for feature in ls_cols_original
                if feature in dict_reconstructions.keys()
            ],
        ]
        df_reconstructions.rename(
            columns={
                column: column + "_recon" for column in df_reconstructions.columns
            },
            inplace=True,
        )
        return df_reconstructions

    @classmethod
    def _format_dict_reconstructions(
        cls,
        ls_cat_features: List[str],
        ls_con_features: List[str],
        dict_cat_feature_to_tokenizer: Dict[str, Tokenizer],
        tup_t_cat_reconstructions: Tuple[torch.tensor],
        t_out_con_reconstructions: torch.tensor,
    ) -> Dict[str, Union[List[str], List[float]]]:
        dict_reconstructions = {}
        for i, cat_feature in enumerate(ls_cat_features):
            dict_reconstructions[
                cat_feature
            ] = cls._get_cat_feature_reconstruction_from_tensor(
                t_cat_reconstruction=tup_t_cat_reconstructions[i],
                tokenizer=dict_cat_feature_to_tokenizer[cat_feature],
            )
        for i, con_feature in enumerate(ls_con_features):
            dict_reconstructions[
                con_feature
            ] = cls._get_con_feature_reconstruction_from_tensor(
                t_con_reconstruction=t_out_con_reconstructions[:, i]
            )
        return dict_reconstructions

    @staticmethod
    def _get_cat_feature_reconstruction_from_tensor(
        t_cat_reconstruction: torch.tensor, tokenizer: Tokenizer
    ) -> List[str]:
        ls_cat_reconstruction = t_cat_reconstruction.tolist()
        ls_cat_reconstruction = tokenizer.decode(ls_cat_reconstruction)
        return ls_cat_reconstruction

    @staticmethod
    def _get_con_feature_reconstruction_from_tensor(
        t_con_reconstruction: torch.tensor,
    ) -> List[float]:
        ls_con_reconstruction = t_con_reconstruction.tolist()
        return ls_con_reconstruction

    @staticmethod
    def _format_sr_loss_by_record(
        index: List[Any],
        t_reconstruction_loss: torch.tensor,  # t_reconstruction_loss shape: (B)
    ) -> pd.Series:
        sr_loss_by_record = pd.Series(data=t_reconstruction_loss.tolist(), index=index)
        return sr_loss_by_record

    @staticmethod
    def _format_sr_loss_by_feature(dict_loss_by_feature: Dict[str, float]) -> pd.Series:
        sr_loss_by_feature = pd.Series(dict_loss_by_feature)
        return sr_loss_by_feature.sort_values(ascending=False)

    @staticmethod
    def _get_dict_loss_by_feature(
        ls_features: List[str],
        t_losses: torch.tensor,  # Shape (n_cat_features + n_con_features)
    ) -> Dict[str, float]:
        dict_loss_by_feature = {}
        for i, feature in enumerate(ls_features):
            dict_loss_by_feature[feature] = t_losses[i].item()
        return dict_loss_by_feature

    @staticmethod
    @torch.no_grad()
    def _get_reconstruction_tensors(
        autoencoder: Autoencoder,
        t_input_data: torch.tensor,  # Shape: (B, n_cat_features + n_con_features)
        argmax_cat_logits: bool,
        denormalize_con_outputs: bool,
    ) -> Tuple[Optional[Tuple[torch.tensor]], Optional[torch.tensor]]:
        _, tup_t_out_cat, t_out_con, _, _, _ = autoencoder.forward(
            t_in=t_input_data,
            compute_loss=False,
            argmax_cat_logits=argmax_cat_logits,
            denormalize_con_outputs=denormalize_con_outputs,
        )
        return tup_t_out_cat, t_out_con

    @staticmethod
    @torch.no_grad()
    def _get_latent_rep_tensor(
        autoencoder: Autoencoder,
        t_input_data: torch.tensor,  # Shape: (B, n_cat_features + n_con_features)
    ) -> torch.tensor:
        t_latent_rep, _, _, _, _, _ = autoencoder(
            t_in=t_input_data,
            compute_loss=False,
        )
        return t_latent_rep

    @staticmethod
    @torch.no_grad()
    def _get_loss_tensors(
        autoencoder: Autoencoder,
        t_input_data: torch.tensor,  # Shape: (B, n_cat_features + n_con_features)
        loss_batch_reduction: str,  # 'none', 'mean', or 'sum'
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        _, _, _, t_loss, t_cat_losses, t_con_losses = autoencoder.forward(
            t_in=t_input_data,
            compute_loss=True,
            loss_batch_reduction=loss_batch_reduction,
        )
        return t_loss, t_cat_losses, t_con_losses

    # Shapes: (B), (B, n_cat_features), (B, n_con_features) if no reduction is applied (loss_batch_reduction = 'none').
    # Shapes: (), (n_cat_features), (n_con_features) if reduction is applied (loss_batch_reduction = 'mean' or 'sum').

    @classmethod
    def _prepare_t_input(
        cls,
        input_data: Union[pd.Series, pd.DataFrame],
        ls_cat_features: List[str],
        ls_con_features: List[str],
        dict_cat_feature_to_tokenizer: Dict[str, Tokenizer],
    ) -> torch.tensor:
        if type(input_data) == pd.Series:
            df_prepared_data = cls._sr_to_df(
                sr=input_data,
                ls_cat_features=ls_cat_features,
                ls_con_features=ls_con_features,
            )
        else:
            df_prepared_data = input_data.copy()
        for cat_feature, tokenizer in dict_cat_feature_to_tokenizer.items():
            df_prepared_data[cat_feature] = tokenizer.encode(
                list(df_prepared_data[cat_feature].values)
            )
        ls_ordered_feature_names = cls._get_ordered_feature_names(
            ls_cat_features=ls_cat_features, ls_con_features=ls_con_features
        )
        df_prepared_data = df_prepared_data.loc[:, ls_ordered_feature_names]
        t_input = torch.tensor(df_prepared_data.values)
        return t_input

    @classmethod
    def _sr_to_df(
        cls, sr: pd.Series, ls_cat_features: List[str], ls_con_features: List[str]
    ) -> pd.DataFrame:
        schema = cls._get_schema(
            ls_cat_features=ls_cat_features,
            ls_con_features=ls_con_features,
        )
        df = pd.DataFrame(sr).T
        df = df.astype(dtype=schema)
        return df

    @staticmethod
    def _get_schema(
        ls_cat_features: List[str], ls_con_features: List[str]
    ) -> Dict[str, type]:
        schema = {}
        for cat_feature in ls_cat_features:
            schema[cat_feature] = object
        for con_feature in ls_con_features:
            schema[con_feature] = np.float64
        return schema

    @staticmethod
    def _get_ordered_feature_names(
        ls_cat_features: List[str], ls_con_features: List[str]
    ) -> List[str]:
        return ls_cat_features + ls_con_features

    @staticmethod
    def _extract_feature_names(input_data: Union[pd.Series, pd.DataFrame]) -> List[str]:
        if type(input_data) == pd.Series:
            return input_data.index.tolist()
        elif type(input_data) == pd.DataFrame:
            return input_data.columns.tolist()

    @staticmethod
    def _get_dict_cat_feature_to_tokenizer(
        dict_cat_feature_to_ls_categories: Dict[str, List[str]]
    ) -> Dict[str, Tokenizer]:
        dict_cat_feature_to_tokenizer = {
            cat_feature: Tokenizer(ls_tokens=ls_categories)
            for cat_feature, ls_categories in dict_cat_feature_to_ls_categories.items()
        }
        return dict_cat_feature_to_tokenizer