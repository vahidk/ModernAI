import typing

import tiktoken
import torch

BASE_ENCODING = "gpt2"


class SpecialTokens:
    BEGIN = "<|begin|>"
    END = "<|end|>"


def get_encoding() -> tiktoken.Encoding:
    base_enc = tiktoken.get_encoding(BASE_ENCODING)
    offset = base_enc.max_token_value + 1
    encoding = tiktoken.Encoding(
        name="gpt2",
        pat_str=base_enc._pat_str,
        mergeable_ranks=base_enc._mergeable_ranks,
        special_tokens={
            SpecialTokens.BEGIN: offset,
            SpecialTokens.END: offset + 1,
        },
    )
    return encoding


def decode_batch(encoding: tiktoken.Encoding, tokens: torch.Tensor, trim: bool = False) -> typing.Generator:
    for seq in tokens:
        output = encoding.decode(seq.tolist())
        if trim:
            output = output.split(SpecialTokens.BEGIN)[1]
            output = output.split(SpecialTokens.END)[0]
        yield output
