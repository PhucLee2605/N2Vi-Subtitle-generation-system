from utils.model import enhance_pipeline
from utils.preprocess import map_to_array
import torch
import config_with_yaml as config
import os
import argparse

cfg = config.load(r"utils/config.yaml")
chunk_length = cfg.getProperty("chunk_lenght_s")


def speech_recognize(audio_file: str):
    assert os.path.isfile(audio_file), f"{audio_file} not exists"
    ds = map_to_array({
        "file": audio_file
    })

    with torch.no_grad():
        transcription = enhance_pipeline()(
            ds["speech"], chunk_length_s=chunk_length)["text"]

    return transcription

#TODO: test
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True,
                        help="file for recognize speech")
    args = parser.parse_args()
    audio_file = args.file

    print(f"Result of speech recognition:\n\"{speech_recognize(audio_file)}\"")
