"""Configs for LightningModules."""
from hsdl.config.base import Config


class TransformerConfig(Config):

    def __init__(self,
                 model_name: str,
                 num_classes: int,
                 max_seq_length: int = 128):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_seq_length = max_seq_length
