from nltk.translate.bleu_score import sentence_bleu
from .model import backBone, infer
import torch


def bleuScore(sen1, sen2):
    words1 = sen1.split(' ')
    words2 = sen2.split(' ')
    bleuscore = sentence_bleu([words1], words2)

    return bleuscore


def valuateTranslate(data, max_length=256):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    tokenizer, model = backBone(device)
    print('Done model')
    predict = infer(data, tokenizer, model, max_length, device)

    score = sum([bleuScore(can, infer) for can, infer in predict]) / len(predict)
    return score
