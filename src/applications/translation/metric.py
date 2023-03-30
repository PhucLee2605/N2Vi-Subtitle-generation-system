from typing import List
from nltk.translate.bleu_score import sentence_bleu
from .model import translate_model, infer
import torch

#TODO: complete docstring
def cal_bleu(sen1: str, sen2: str) -> float:
    """_summary_

    Args:
        sen1 (str): _description_
        sen2 (str): _description_

    Returns:
        float: _description_
    """
    words1 = sen1.split(' ')
    words2 = sen2.split(' ')
    bleuscore = sentence_bleu([words1], words2)

    return bleuscore


#TODO: complete docstring
def valuate_translation(data: List[str], ground_truth: List[str], max_length: int = 256) -> float:
    """_summary_

    Args:
        data (List[str]): _description_
        max_length (int, optional): _description_. Defaults to 256.

    Returns:
        float: _description_
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer, model = translate_model(device)
    prediction = infer(data, tokenizer, model, max_length, device)

    score = sum([cal_bleu(prediction[index], ground_truth[index])
                for index in range(len(prediction))]) / len(prediction)
    return score
