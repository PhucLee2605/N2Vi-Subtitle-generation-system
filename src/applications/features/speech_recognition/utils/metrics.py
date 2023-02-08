from evaluate import load
from typing import List


def cal_average_WER(predictions: List[str], references: List[str]):
    assert len(predictions) == len(
        references), "predictions and references must have the same number of elements"
    wer_metric = load("wer")
    total_wer = 0

    for i in range(len(predictions)):
        total_wer += wer_metric.compute(predictions=predictions[i],
                                        references=references[i])

    return total_wer/len(predictions)


def cal_single_WER(prediction: str, reference: str):
    assert len(prediction) == len(
        reference), "predictions and references must have the same number of elements"

    if len(prediction) > 1:
        print("More than 1 element, redirecting to calculating average WER function")
        return cal_average_WER(prediction, reference)

    wer_metric = load("wer")
    return wer_metric.compute(predctions=prediction, references=reference)