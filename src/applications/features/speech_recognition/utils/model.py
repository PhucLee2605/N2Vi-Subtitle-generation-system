from transformers import pipeline
# from .util import cfg, DEVICE
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# device = -1 if DEVICE == "cpu" else 0


def enhance_pipeline(name: str = None):
    # if not name:
    #     name = cfg.getProperty("model_name")
    return pipeline(tokenizer=name, model=name, device=-1, return_timestamps="word")


def tokenizer(tokenizer_name: str = None):
    # if not tokenizer_name:
    #     tokenizer_name = cfg.getProperty("model_name")
    return Wav2Vec2Processor.from_pretrained(tokenizer_name)


def Wav2Vec2(model_name: str = None):
    # if not model_name:
    #     model_name = cfg.getProperty("model_name")
    return Wav2Vec2ForCTC.from_pretrained(model_name)
