import torch
from typing import Any, Union
from .recognition.model import enhance_pipeline
import config_with_yaml as config


cfg = config.load("src/applications/recognition/recog_config.yaml")

CHUNK_LENGTH = cfg.getProperty("chunk_lenght_s")


def speech_recognize(audio: Union[str, Any], recog_model, chunk_length=CHUNK_LENGTH, sampling_rate: int = 22500) -> dict:
    """Do speech recognition and return both raw text and words with timestamps

    Args:
        audio (Union[str,Any]): can be file's name or audio loaded by librosa (mono)
        enhance (boll): set to True to enhance speech before pass into recognition model (set to False if already enhanced or speech's quality is good)
    Returns:
        _type_: {"text": "raw text from recognition",
                "chunks": [
                            {"text": "word1", "timestamps": (start, end)},
                            {"text": "word2", "timestamps": (start, end)},
                            ...
                        ]
                }
    """
    ds = dict()

    ds["speech"] = audio
    ds["sampling rate"] = sampling_rate

    # if enhance:
    #     ds["speech"], _ = enhance_speech(ds["speech"],
    #                                      sampling_rate=ds["sampling rate"])

    ds["speech"] = ds["speech"].squeeze().cpu().detach().numpy()

    with torch.no_grad():
        transcription = recog_model(ds["speech"], chunk_length_s=chunk_length)

    print("[INFO] Finished speech recognition")
    return transcription


class Recognition():
    def __init__(self, chunk_length=CHUNK_LENGTH, lang="en", device="cpu"):
        if lang == "vi":
            self.recog_model = enhance_pipeline(cfg.getProperty("vi_model_name"), device)
        else:
            self.recog_model = enhance_pipeline(cfg.getProperty("en_model_name"), device)

        self.chunk_length = chunk_length

    def infer(self, audio):
        return speech_recognize(audio, self.recog_model, self.chunk_length)