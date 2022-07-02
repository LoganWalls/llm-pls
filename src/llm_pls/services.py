import sys

import torch
from torch.nn.functional import log_softmax
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

from . import config
from .model import CompletionParams, get_hf_params


def sanitize_float(x: float) -> float:
    """Sanitize floats to that they are JSON compatibles
    (e.g. inf and -inf are not valid JSON)"""
    if x == float("inf"):
        return sys.float_info.max
    elif x == float("-inf"):
        return -sys.float_info.max
    else:
        return x


class ModelService:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = config.input_device

    def generate_completion(self, params: CompletionParams) -> dict:
        hf_params = get_hf_params(params, self.tokenizer)
        response = self.model.generate(
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            **hf_params
        )
        # Remove the prompt from the response
        response_start = hf_params["inputs"].size(1)
        # We assume batch-size is always 1, so we hard-code 0
        token_ids = response.sequences[0, response_start:].tolist()  # type: ignore
        tokens = [self.tokenizer.decode(t) for t in token_ids]
        result = dict(text="".join(tokens))
        if params.logprobs:
            top_logprobs = []
            token_logprobs = []
            for t_id, s in zip(token_ids, response.scores[response_start:]):  # type: ignore
                lps = log_softmax(s[0], dim=-1)
                token_logprobs.append(sanitize_float(lps[t_id].item()))
                lps, token_ids = lps.topk(params.logprobs)
                top_logprobs.append(
                    {
                        self.tokenizer.decode(t): sanitize_float(l)
                        for t, l in zip(token_ids, lps.tolist())
                    }
                )
            result["logprobs"] = dict(  # type: ignore
                tokens=tokens,
                token_logprobs=token_logprobs,
                top_logprobs=top_logprobs,
            )
        return result
