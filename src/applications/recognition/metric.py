from evaluate import load
from typing import List


def cal_average_WER(predictions: List[str], references: List[str]):
    assert len(predictions) == len(
        references), "predictions and references must have the same number of elements"
    wer = load("wer")
    total_wer = 0

    for i in range(len(predictions)):
        total_wer += wer.compute(predictions=[predictions[i]],
                                 references=[references[i]])

    return total_wer/len(predictions)


def cal_single_WER(prediction: List[str], reference: List[str]):
    assert len(prediction) == len(
        reference), "predictions and references must have the same number of elements"

    if len(prediction) > 1:
        print("More than 1 element, redirecting to calculating average WER function")
        return cal_average_WER(prediction, reference)

    wer = load("wer")
    return wer.compute(predictions=prediction, references=reference)
