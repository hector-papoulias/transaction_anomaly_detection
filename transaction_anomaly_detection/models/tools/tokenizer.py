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
