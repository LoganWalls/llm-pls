from dataclasses import dataclass
from itertools import chain
from typing import List, Mapping, Optional, Tuple, Union

import torch
from pydantic import BaseModel
from transformers.generation_stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from . import config


class CompletionParams(BaseModel):
    """Parameters to control completions.
    See: https://beta.openai.com/docs/api-reference/completion
    """

    prompt: str
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    logprobs: Optional[int] = None
    stop: Optional[Union[List[str], str]] = None
    # presence_penalty: float = 0.0
    # frequency_penalty: float = 0.0
    best_of: int = 1
    # logit_bias: Optional[Mapping[str, float]] = None


class TokenStoppingCriteria(StoppingCriteria):
    def __init__(self, token_ids: List[int], device: Union[int, str]):
        self.device = device
        self.token_ids = torch.tensor(token_ids, device=self.device)  # type: ignore

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        return bool(torch.any(input_ids[:, -1] == self.token_ids))


def get_stopping_criteria(
    stop_strings: Union[str, List[str]], tokenizer: PreTrainedTokenizer
) -> StoppingCriteriaList:
    if isinstance(stop_strings, str):
        stop_strings = [stop_strings]
    stop_token_ids = list(
        chain.from_iterable(tokenizer(stop_strings, add_special_tokens=False).input_ids)
    )
    return StoppingCriteriaList(
        [TokenStoppingCriteria(stop_token_ids, config.input_device)]
    )


def get_hf_params(params: CompletionParams, tokenizer: PreTrainedTokenizer) -> dict:
    hf_params = dict(
        inputs=tokenizer(params.prompt, return_tensors="pt").input_ids.to(
            config.input_device
        ),
        prompt=params.prompt,
        max_new_tokens=params.max_tokens,
        temperature=params.temperature,
        do_sample=params.temperature > 0.0,
        top_p=params.top_p,
        top_k=None,
        num_return_sequences=params.n,
        output_scores=bool(params.logprobs),
        num_beams=params.best_of,
    )
    if params.stop:
        hf_params["stopping_criteria"] = get_stopping_criteria(params.stop, tokenizer)  # type: ignore
    return hf_params


def load_model(model_name: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    kwargs = dict()
    if config.input_device != "cpu":
        kwargs = dict(device_map="auto", torch_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model, tokenizer
