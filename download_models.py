"""Script to download required transformer model parameters."""
import os

# from ckiptagger import data_utils
from transformers import AutoModel, AutoTokenizer


from_transformers = [
    'bert-base-chinese',
    'roberta-base',
    'distilbert-base-uncased',
    'distilbert-base-multilingual-cased',
    'xlm-roberta-base',
    'hfl/chinese-roberta-wwm-ext',
]


if __name__ == '__main__':
    # print('Downloading CkipTagger...')
    # if not os.path.exists('/ckip'):
    #     os.mkdir('/ckip')
    # data_utils.downlaod_data_gdown('/ckip')
    print('Downloading models...')
    for model_name in from_transformers:
        _ = AutoTokenizer.from_pretrained(model_name)
        _ = AutoModel.from_pretrained(model_name)
    print('Models downloaded successfully.')
