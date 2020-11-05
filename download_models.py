"""Script to download required transformer model parameters."""
from transformers import AutoModel, AutoTokenizer


from_transformers = [
    'roberta-base',
    'distilbert-base-uncased',
    'distilbert-base-multilingual-cased',
    'xlm-roberta-base',
    'hfl/chinese-roberta-wwm-ext',
]


if __name__ == '__main__':
    print('Downloading models...')
    for model_name in from_transformers:
        _ = AutoTokenizer.from_pretrained(model_name)
        _ = AutoModel.from_pretrained(model_name)
    print('Models downloaded successfully.')
