from transformers import pipeline
import config_with_yaml as config
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

cfg = config.load("utils/config.yaml")
device = -1 if cfg.getProperty("device") == "cpu" else 0


def enhance_pipeline(name: str = None):
    if not name:
        name = cfg.getProperty("model_name")
    return pipeline(tokenizer=name, model=name, device=device)


def tokenizer(tokenizer_name: str = None):
    if not tokenizer_name:
        tokenizer_name = cfg.getProperty("model_name")
    return Wav2Vec2Processor.from_pretrained(tokenizer_name)


def Wav2Vec2(model_name: str = None):
    if not model_name:
        model_name = cfg.getProperty("model_name")
    return Wav2Vec2ForCTC.from_pretrained(model_name)

