from torch.nn.functional import log_softmax
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

from .model import CompletionParams, get_hf_params


class ModelService:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = "cpu"

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
                token_logprobs.append(lps[t_id].item())
                lps, token_ids = lps.topk(params.logprobs)
                top_logprobs.append(
                    {
                        self.tokenizer.decode(t): l
                        for t, l in zip(token_ids, lps.tolist())
                    }
                )
            result["logprobs"] = dict(  # type: ignore
                tokens=tokens,
                token_logprobs=token_logprobs,
                top_logprobs=top_logprobs,
            )
            print(result)
        return result
