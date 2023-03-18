import os
from typing import Any
import config_with_yaml as config
from transformers import pipeline
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, WhisperProcessor, WhisperForConditionalGeneration


cfg = config.load(os.path.join(os.path.dirname(__file__),"recog_config.yaml"))


def recog_pipeline(name: str = None, device="cpu") -> pipeline:
    """ create enhnacement pipeline with pretrained model

    Args:
        name (str, optional): model's name that will be used. Defaults to None.
        device (str, optional): device that pipeline will be run on. Defaults to "cpu".

    Returns:
        pipeline: transformers pipeline of pretrained model and tokenizer
    """
    if not name:
        name = cfg.getProperty("model_name")
    return pipeline("automatic-speech-recognition",tokenizer=name, model=name, device=device)


def tokenizer(tokenizer_name: str = None) -> Wav2Vec2Processor:
    """ create tokenizer from pretrained Wav2Vec2 tokenizer

    Args:
        tokenizer_name (str, optional): pretrained tokenizer name. Defaults to None.

    Returns:
        Wav2Vec2Processor: pretrained Wav2Vec2 tokenizer
    """
    if not tokenizer_name:
        tokenizer_name = cfg.getProperty("model_name")
    return Wav2Vec2Processor.from_pretrained(tokenizer_name)


def Wav2Vec2(model_name: str = None) -> Wav2Vec2ForCTC:
    """ Create Wav2Vec2 model from pretrained model's name

    Args:
        model_name (str, optional): pretrained model's name. Defaults to None.

    Returns:
        Wav2Vec2ForCTC: Wav2Vec2 model with pretrained weight
    """
    if not model_name:
        model_name = cfg.getProperty("model_name")
    return Wav2Vec2ForCTC.from_pretrained(model_name)
