"""Probability of a sentence via BERT.

https://arxiv.org/pdf/2004.00881.pdf
"""
import math

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


def prob(sent: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) \
        -> float:
    # make sure to do this before calling this:
    #     model = model.cuda()
    #     model.eval()

    # get outputs
    input_ixs = tokenizer([sent], return_tensors='pt')['input_ids'].cuda()

    log_probs = []

    # skip the first ([CLS]) and last ([SEP])
    for ix in range(1, input_ixs.shape[1] - 1):
        input_ixs[0, ix] = tokenizer.mask_token_id
        with torch.no_grad():
            logits = model(input_ixs)['logits']
        word_ix = input_ixs[0, ix]
        preds = logits[0, ix]
        preds = torch.softmax(preds, dim=0)
        pred = float(preds[word_ix].cpu().numpy())
        log_prob = math.log(pred)
        log_probs.append(log_prob)

    return sum(log_probs) / input_ixs.shape[1]
