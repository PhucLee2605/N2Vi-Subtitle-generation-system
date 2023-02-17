from .utils.model import enhance_pipeline
from .utils.preprocess import map_to_array
from .utils.util import cfg, export_xml
import torch
import os
from typing import Any, Union
import argparse
import librosa as ls

chunk_length = cfg.getProperty("chunk_lenght_s")


def speech_recognize(audio: Union[str, Any]):
    """Do speech recognition and return both raw text and words with timestamps

    Args:
        audio (Union[str,Any]): can be file's name or audio loaded by librosa (mono)

    Returns:
        _type_: {"text": "raw text from recognition",
                "chunk": [
                            {"text": "word1", "timestamps": (start, end)},
                            {"text": "word2", "timestamps": (start, end)},
                            ...
                        ]        
                }
    """
    ds = dict()
    try:
        assert os.path.isfile(audio)
        ds = map_to_array({
            "file": audio
        })
    except AssertionError:
        ds["speech"] = audio

    with torch.no_grad():
        transcription = enhance_pipeline()(
            ds["speech"], chunk_length_s=chunk_length)
    export_xml(transcription)

    return transcription


#TODO: test
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True,
                        help="file for recognize speech")
    args = parser.parse_args()
    audio_file = args.file

    print(f"Result of speech recognition:\n\"{speech_recognize(audio_file)}\"")
