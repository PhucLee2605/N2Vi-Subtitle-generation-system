import torch
from typing import Any, Callable, Union
import config_with_yaml as config
from denoiser.dsp import convert_audio

from .recognition.model import recog_pipeline

cfg = config.load("src/applications/recognition/recog_config.yaml")

CHUNK_LENGTH = cfg.getProperty("chunk_lenght_s")


def speech_recognize(audio: Any, recog_model: Any, chunk_length: int = CHUNK_LENGTH, sampling_rate: int = 16000, device: str = "cpu") -> dict:
    """ Do speech recognition and return both raw text and sentences with timestamps

    Args:
        audio (Any): Loaded audio in form of tensor
        recog_model (Any): _description_
        chunk_length (int, optional): Split input audio into chunks to infer model with big tensor without crashing memory. Defaults to CHUNK_LENGTH.
        sampling_rate (int, optional): input audio's sampling rate. Defaults to 16000Hz.
        device (str, optional): device to run speech recognition on. Defaut to "cpu"

    Returns:
        dict: {"text": "raw text from recognition",
                "chunks": [
                            {"text": "sentence1", "timestamps": (start, end)},
                            {"text": "sentence2", "timestamps": (start, end)},
                            ...
                        ]
                }
    """

    ds = dict()
    audio = convert_audio(wav=audio.to(device),
                          from_samplerate=sampling_rate,
                          to_samplerate=16000,
                          channels=1)
    ds["speech"] = audio
    ds["sampling rate"] = sampling_rate

    ds["speech"] = ds["speech"].squeeze().cpu().detach().numpy()

    with torch.no_grad():
        transcription = recog_model(ds["speech"],
                                    chunk_length_s=chunk_length,
                                    return_timestamps=True)

    print("[INFO] Finished speech recognition")
    return transcription


class Recognition():
    def __init__(self, chunk_length: int = CHUNK_LENGTH, lang: str = "en", device: str = "cpu") -> None:
        """ Initiate Recognition

        Args:
            chunk_length (int, optional): Split input audio into chunks to infer model with big tensor without crashing memory. Defaults to CHUNK_LENGTH.
            lang (str, optional): Language of input audio to recognize. "en" for English, "vi" for Vietnamses. Defaults to "en" - English.
            device (str, optional): device to run inference on. Defaults to "cpu".
        """
        if lang == "vi":
            self.recog_model = recog_pipeline(cfg.getProperty("vi_model_name"),
                                              device)
        elif lang == "en":
            self.recog_model = recog_pipeline(cfg.getProperty("en_model_name"),
                                              device)
        self.device = device
        self.chunk_length = chunk_length

    def infer(self, audio: Any, sampling_rate: int = 16000) -> Callable[[], dict]:
        """ Infering speech recognition

        Args:
            audio (Any): Loaded audio to infer
            sampling_rate (int): input audio's sampling rate. Default to 16000

        Returns:
            Callable[[], dict]: speech_recognize function
        """
        return speech_recognize(audio=audio, recog_model=self.recog_model, chunk_length=self.chunk_length, device=self.device, sampling_rate=sampling_rate)
