"""Utility code for working with pytorch_transformers."""
import torch
from torch.utils import data as td
import transformers as tf

from arct import util
from arct.util import training


class SingleInputExample(object):

    def __init__(self, guid, text_a, label):
        self.guid = guid
        self.text_a = text_a
        self.text_b = None
        self.label = label

    def __repr__(self):
        info = 'Guid:  %s\n' % self.guid
        info += 'Text:  %s\n' % self.text_a
        info += 'Label: %s' % self.label
        return info


class InputExample(object):
    # note: always two sents - concat claim and premise as sent 1.

    def __init__(self, guid, text_a, text_b, label):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        info = 'Guid:    %s\n' % self.guid
        info += 'Text A:  %s\n' % self.text_a
        info += 'Text b:  %s\n' % self.text_b
        info += 'Label:   %s' % self.label
        return info


class InputFeatures(object):

    def __init__(self, input_ids, attention_mask, tokens, label_id):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.tokens = tokens
        self.label_id = label_id


class DataProcessor(object):

    def get_train_examples(self):
        raise NotImplementedError()

    def get_dev_examples(self):
        raise NotImplementedError()

    def get_test_examples(self):
        raise NotImplementedError

    def get_labels(self):
        raise NotImplementedError()


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            truncate_seq_pair(tokens_a, tokens_b,
                              max_seq_length - tokenizer.n_specials)
        else:
            if len(tokens_a) > max_seq_length - tokenizer.n_specials:
                tokens_a = tokens_a[0:(max_seq_length - tokenizer.n_specials)]

        input_ids_0 = tokenizer.convert_tokens_to_ids(tokens_a)
        input_ids_1 = tokenizer.convert_tokens_to_ids(tokens_b) \
            if tokens_b else None

        input_ids = tokenizer.build_inputs(input_ids_0, input_ids_1)
        attention_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(tokenizer.pad_id)
            attention_mask.append(0)
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        tokens = [util.pad if t == tokenizer.pad_token else t
                  for t in tokens]

        label_id = label_map[example.label] \
            if example.label is not None else None

        assert len(input_ids) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          tokens=tokens,
                          label_id=label_id))

    return features


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal
    # percent of tokens from each, since if one sequence is very short then each
    # token that's truncated likely contains more information than a longer
    # sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


#
# Tokenizers


class TransformerTokenizer:

    def __init__(self, tokenizer_type, config):
        self.add_prefix_space = config.model_type in ['gpt2', 'roberta']
        self.tokenizer = tokenizer_type.from_pretrained(config.model_name)
        if config.model_type == 'gpt2':
            _ = self.tokenizer.add_special_tokens({'pad_token': ';'})
        self.config = config
        self.n_specials = self.get_n_specials()

    def build_inputs(self, ids_1, ids_2=None):
        return self.tokenizer.build_inputs_with_special_tokens(ids_1, ids_2)

    def check_input(self, text1, text2):
        tokens_1 = self.tokenize(text1)
        tokens_2 = self.tokenize(text2)
        ids_1 = self.convert_tokens_to_ids(tokens_1)
        ids_2 = self.convert_tokens_to_ids(tokens_2)
        inputs = self.build_inputs(ids_1, ids_2)
        return ' '.join(self.convert_ids_to_tokens(inputs))

    def convert_ids_to_tokens(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids)

    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def get_n_specials(self):
        text_a = 'one two three'
        text_b = 'four five six'
        tokens_a = self.tokenize(text_a)
        assert len(tokens_a) == 3
        tokens_b = self.tokenize(text_b)
        assert len(tokens_b) == 3
        ids_a = self.convert_tokens_to_ids(tokens_a)
        ids_b = self.convert_tokens_to_ids(tokens_b)
        input_ids = self.tokenizer.build_inputs_with_special_tokens(
            ids_a, ids_b)
        return len(input_ids) - 6

    def tokenize(self, text):
        if self.config.model_type in ['gpt2', 'roberta']:
            return self.tokenizer.tokenize(
                text, add_prefix_space=self.add_prefix_space)
        else:
            return self.tokenizer.tokenize(text)

    @property
    def pad_id(self):
        return self.tokenizer.pad_token_id

    @property
    def pad_token(self):
        return self.tokenizer.pad_token
