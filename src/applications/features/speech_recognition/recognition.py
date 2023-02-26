import sys
import os
sys.path.insert(0, f'{os.path.dirname(__file__)}/..')
import torch
import argparse
from typing import Any, Union, Literal

from speech_enhancement.enhancement import enhance_speech
from .utils.model import enhance_pipeline
from .utils.preprocess import load_audio_file
from .utils.util import cfg, export_xml, model_en, model_vi


chunk_length = cfg.getProperty("chunk_lenght_s")


def speech_recognize(audio: Union[str, Any], sampling_rate : int = 22500, enhance: bool = True, lang: Literal["vi", "en"] = "en") -> dict:
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
    try:
        assert os.path.isfile(audio)
        ds = load_audio_file({
            "file": audio
        })
    except (AssertionError, TypeError):
        ds["speech"] = audio
        ds["sampling rate"] = sampling_rate

    if enhance:
        ds["speech"], _ = enhance_speech(ds["speech"],
                                         sampling_rate=ds["sampling rate"])
    
    ds["speech"] = ds["speech"].squeeze().cpu().detach().numpy()

    with torch.no_grad():
        if lang == "en":
            transcription = model_en(ds["speech"], chunk_length_s=chunk_length)
        elif lang == "vi":
            transcription = model_vi(ds["speech"], chunk_length_s=chunk_length)

    print("[INFO] Finished speech recognition")
    return transcription


#TODO: test
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True,
                        help="file for recognize speech")
    args = parser.parse_args()