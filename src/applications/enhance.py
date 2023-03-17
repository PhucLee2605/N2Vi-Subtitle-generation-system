import torch
from typing import Any, Callable, Tuple, Union, Literal
from denoiser.dsp import convert_audio
import config_with_yaml as config

from .enhancement.model import denoise_model

cfg = config.load("src/applications/enhancement/enhance_config.yaml")


def enhance_speech(audio: Any, model: Any, sampling_rate: int = 16000, device: str = "cpu") -> Tuple[Any, int]:
    """ Enhance loaded speech, return denoised, vocal enhanced speech and sampling rate

    Args:
        audio (Any): loaded speech in form of torch tensor
        model (Any): model used to enhance speech. Currently support pretrained model of Facebook's denoiser
        sampling_rate (int, optional): sampling rate of input speech. Defaults to 16000.
        device (str, optional): Device used to run enhancement on. Defaults to "cpu".

    Returns:
        Tuple[Any, int]: Tuple of enhanced speech and sampling rate
    """

    converted_audio = convert_audio(wav=audio.to(device),
                                    from_samplerate=sampling_rate,
                                    to_samplerate=model.sample_rate,
                                    channels=model.chin)

    with torch.no_grad():
        denoised_audio = model(converted_audio[None])[0]
        denoised_audio = torch.mean(denoised_audio, dim=0, keepdim=True)

    print("[INFO] Finished speech enhancement")
    return (denoised_audio, model.sample_rate)


class Enhancement():
    def __init__(self, model: Literal["dns48", "dns64", "master64"] = None, device: str = "cpu") -> None:
        """ Initiate Enhancement object

        Args:
            model (Literal['dns48', 'dns64', 'master64'], optional): Model name will be used to enhance speech. Defaults to None.
            device (str, optional): Device that will run enhancement on. Defaults to "cpu".
        """
        self.device = device
        if isinstance(model, str):
            model = denoise_model(model)
        else:
            model = denoise_model(cfg.getProperty("model"))
        self.model = model.to(device)

    def infer(self, audio: Any, sr: int = 16000) -> Callable[[], Tuple[Any, int]]:
        """ Inference of speech enhancement

        Args:
            audio (Any): loaded audio to be enhanced
            sr (int, optional): input audio sampling rate (in Hz). Defaults to 16000.

        Returns:
            Callable[[], Tuple[Any, int]]: enhancement function
        """
        return enhance_speech(audio, model=self.model, sampling_rate=sr, device=self.device)
