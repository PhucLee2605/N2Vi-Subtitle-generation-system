from evaluate import load
from typing import List


def cal_wer_score(prediction: List[str], reference: List[str]) -> float:
    """ Calculate word error rate between predicitons and references

    Args:
        prediction (List[str]): List of predictions string 
        reference (List[str]): List of ground truth string

    Returns:
        float: WER score
    """
    wer = load("wer")
    return wer.compute(predictions=prediction, references=reference)
