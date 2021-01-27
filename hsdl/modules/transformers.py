from typing import Dict, Tuple, Union

from hsdl.experiments.config import ExperimentConfig
from hsdl.modules import BaseModule
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from transformers import AutoModel


class Transformer(BaseModule):

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.transformer = AutoModel.from_pretrained(config.model.model_name)
        self.dropout = nn.Dropout(p=config.training.dropout)
        self.classify = nn.Linear(768, config.model.num_classes)

    def add_metric(self,
                   logits: Tensor,
                   y: Union[Tensor, Tuple],
                   subset: str):
        y, _ = y
        return self.metrics[subset](logits, y)

    def forward(self, x: Tensor) -> Tensor:
        h = self.transformer(x).last_hidden_state.sum(axis=1)
        h = self.dropout(h)
        logits = self.classify(h)
        return logits

    def loss(self, logits: Tensor, y: Tuple[Tensor, Tensor]) -> Tensor:
        y, weights = y
        return F.binary_cross_entropy_with_logits(logits, y, weight=weights)

    def predict(self, sent: str, collate, cuda=True) -> Dict:
        self.eval()
        with torch.no_grad():
            batch = [(sent, 0)]
            x, _ = collate(batch, do_labels=False)
            if cuda:
                x = x.cuda()
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            return probs
