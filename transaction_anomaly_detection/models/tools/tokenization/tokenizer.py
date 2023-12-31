from typing import Set, List, Tuple, Dict, Union, Optional, overload


class Tokenizer:
    def __init__(
        self,
        ls_tokens: List[str],
        pad_token: Optional[str] = None,
        unk_token: Optional[str] = None,
        mask_token: Optional[str] = None,
    ):
        # Form regular tokens
        ls_regular_tokens = ls_tokens

        # Form special tokens
        ls_special_tokens = []
        self._pad_token = pad_token
        self._pad_token_encoding = None
        if pad_token is not None:
            ls_special_tokens.append(pad_token)
        self._unk_token = unk_token
        self._unk_token_encoding = None
        if unk_token is not None:
            ls_special_tokens.append(unk_token)
        self._mask_token = mask_token
        self._mask_token_encoding = None
        if mask_token is not None:
            ls_special_tokens.append(mask_token)

        # Form Vocabulary
        ls_vocabulary = ls_special_tokens + ls_regular_tokens

        # Save encoding dicts
        self._encoding_to_token = dict(enumerate(ls_vocabulary))
        self._token_to_encoding = {
            token: encoding for encoding, token in self._encoding_to_token.items()
        }

        # Save vocabulary
        self._regular_tokens = set(ls_regular_tokens)
        self._special_tokens = set(ls_special_tokens)
        self._vocabulary = set(ls_vocabulary)

        # Save encoding values
        self._regular_token_encodings = set(
            [
                self._token_to_encoding[regular_token]
                for regular_token in self._regular_tokens
            ]
        )
        if self._pad_token is not None:
            self._pad_token_encoding = self._token_to_encoding[self._pad_token]
        if self._unk_token is not None:
            self._unk_token_encoding = self._token_to_encoding[self._unk_token]
        if self._mask_token is not None:
            self._mask_token_encoding = self._token_to_encoding[self._mask_token]

    @property
    def vocabulary(self) -> Set[str]:
        return self._vocabulary

    @property
    def pad_token(self) -> Optional[str]:
        return self._pad_token

    @property
    def pad_token_encoding(self) -> Optional[int]:
        return self._pad_token_encoding

    @property
    def unk_token(self) -> Optional[str]:
        return self._unk_token

    @property
    def unk_token_encoding(self) -> Optional[int]:
        return self._unk_token_encoding

    @property
    def mask_token(self) -> Optional[str]:
        return self._mask_token

    @property
    def mask_token_encoding(self) -> Optional[int]:
        return self._mask_token_encoding

    @property
    def regular_token_encodings(self) -> Set[int]:
        return self._regular_token_encodings

    def pad(
        self,
        sequence: List[str],
        pad_left: Optional[int] = 0,
        pad_right: Optional[int] = 0,
    ) -> List[str]:
        if pad_left < 0 or pad_right < 0:
            raise ValueError("pad_left and pad_right should be non-negative integers")
        return self._pad(
            sequence=sequence,
            pad_token=self._pad_token,
            pad_left=pad_left,
            pad_right=pad_right,
        )

    @overload
    def encode(self, token_or_ls_tokens: str) -> int:
        ...

    @overload
    def encode(self, token_or_ls_tokens: List[str]) -> List[int]:
        ...

    def encode(self, token_or_ls_tokens):
        if isinstance(token_or_ls_tokens, list):
            return self._encode_ls_tokens(
                ls_tokens=token_or_ls_tokens,
                token_to_encoding=self._token_to_encoding,
            )
        if isinstance(token_or_ls_tokens, tuple):
            return self._encode_ls_tokens(
                ls_tokens=token_or_ls_tokens,
                token_to_encoding=self._token_to_encoding,
            )
        if isinstance(token_or_ls_tokens, str):
            return self._encode_token(
                token=token_or_ls_tokens,
                token_to_encoding=self._token_to_encoding,
            )

    @overload
    def decode(self, encoded_token_or_ls_encoded_tokens: int) -> str:
        ...

    @overload
    def decode(self, encoded_token_or_ls_encoded_tokens: List[int]) -> List[str]:
        ...

    def decode(self, encoded_token_or_ls_encoded_tokens):
        if isinstance(encoded_token_or_ls_encoded_tokens, list):
            return self._decode_ls_encoded_tokens(
                ls_encoded_tokens=encoded_token_or_ls_encoded_tokens,
                encoding_to_token=self._encoding_to_token,
            )
        if isinstance(encoded_token_or_ls_encoded_tokens, tuple):
            return self._decode_ls_encoded_tokens(
                ls_encoded_tokens=encoded_token_or_ls_encoded_tokens,
                encoding_to_token=self._encoding_to_token,
            )
        if isinstance(encoded_token_or_ls_encoded_tokens, int):
            return self._decode_encoded_token(
                encoded_token=encoded_token_or_ls_encoded_tokens,
                encoding_to_token=self._encoding_to_token,
            )

    def ls_tokens_to_str(self, ls_tokens: List[str]) -> str:
        if self._unk_token is None and not self._tokens_in_vocabulary(
            set_tokens=set(ls_tokens), vocabulary=self._vocabulary
        ):
            raise ValueError("Token not recognized: consider adding an UNK token.")
        else:
            ls_tokens = self._replace_unknown_tokens(
                ls_tokens=ls_tokens,
                vocabulary=self._vocabulary,
                unk_token=self._unk_token,
            )
        return self._ls_tokens_to_str(ls_tokens=ls_tokens)

    def str_to_ls_tokens(self, str_input: str) -> List[str]:
        ls_tokens = self._str_to_ls_tokens(str_input=str_input)
        if self._unk_token is None and not self._tokens_in_vocabulary(
            set_tokens=set(ls_tokens), vocabulary=self._vocabulary
        ):
            raise ValueError("Token not recognized: consider adding an UNK token.")
        else:
            ls_tokens = self._replace_unknown_tokens(
                ls_tokens=ls_tokens,
                vocabulary=self._vocabulary,
                unk_token=self._unk_token,
            )
        return ls_tokens

    @staticmethod
    def _ls_tokens_to_str(
        ls_tokens: List[str],
    ) -> str:
        return " ".join(ls_tokens)

    @staticmethod
    def _str_to_ls_tokens(str_input: str) -> List[str]:
        return list(str_input)

    @classmethod
    def _replace_unknown_tokens(
        cls, ls_tokens: List[str], vocabulary: Set[str], unk_token: str
    ) -> List[str]:
        for i, token in enumerate(ls_tokens):
            if not cls._token_in_vocabulary(token=token, vocabulary=vocabulary):
                ls_tokens[i] = unk_token
        return ls_tokens

    @staticmethod
    def _token_in_vocabulary(token: str, vocabulary: Set[str]) -> bool:
        return token in vocabulary

    @staticmethod
    def _tokens_in_vocabulary(set_tokens: Set[str], vocabulary: Set[str]) -> bool:
        return set_tokens.issubset(vocabulary)

    @classmethod
    def _pad(
        cls,
        sequence: List[str],
        pad_token: str,
        pad_left: int,
        pad_right: int,
    ) -> List[str]:
        return (
            [pad_token for _ in range(pad_left)]
            + sequence
            + [pad_token for _ in range(pad_right)]
        )

    @staticmethod
    def _encode_token(token: str, token_to_encoding: Dict[str, int]) -> int:
        return token_to_encoding[token]

    @classmethod
    def _encode_ls_tokens(
        cls, ls_tokens: Union[List[str], Tuple[str]], token_to_encoding: Dict[str, int]
    ) -> List[int]:
        return list(
            map(
                lambda token: cls._encode_token(
                    token=token, token_to_encoding=token_to_encoding
                ),
                ls_tokens,
            )
        )

    @staticmethod
    def _decode_encoded_token(
        encoded_token: int, encoding_to_token: Dict[int, str]
    ) -> str:
        return encoding_to_token[encoded_token]

    @classmethod
    def _decode_ls_encoded_tokens(
        cls,
        ls_encoded_tokens: Union[List[int], Tuple[int]],
        encoding_to_token: Dict[int, str],
    ) -> List[str]:
        return list(
            map(
                lambda encoded_token: cls._decode_encoded_token(
                    encoded_token=encoded_token, encoding_to_token=encoding_to_token
                ),
                ls_encoded_tokens,
            )
        )
