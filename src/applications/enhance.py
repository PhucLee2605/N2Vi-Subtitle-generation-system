import torch
from typing import Any, Union, Literal
from denoiser.dsp import convert_audio
from .enhancement.model import denoise_model

import config_with_yaml as config

cfg = config.load("src/applications/enhancement/enhance_config.yaml")

def enhance_speech(audio_file: Union[str, Any], model, sampling_rate: int = 22500, device="cpu") -> Union[Any, int]:
    """This funciton enhance speech

    Args:
        audio_file (str, Any): audio's path or loaded ( to be enhanced
        model (Any): type of pretrained model ("dns48", "dns64", "master64"). Already loaded DNS64, specify if want to use another

    Returns:
        Union[Any, int]: return enhanced speecch in the form of torch.Tensor and its sampling rate
    """

    enhanced_audio = convert_audio(wav=audio_file.to(device),
                                   from_samplerate=sampling_rate,
                                   to_samplerate=model.sample_rate,
                                   channels=model.chin)

    with torch.no_grad():
        denoised = model(enhanced_audio[None])[0]
        denoised = torch.mean(denoised, dim=0, keepdim=True)

    print("[INFO] Finished speech enhancement")
    return denoised, sampling_rate


class Enhancement():
    def __init__(self, model: Union[Literal["dns48", "dns64", "master64"], Any] = None, device="cpu"):
        self.device = device
        if isinstance(model, str):
            model = denoise_model(model)
        else:
            model = denoise_model(cfg.getProperty("model"))
        self.model = model.to(device)

    def infer(self, audio, sr=22500):
        return enhance_speech(audio, model=self.model, sampling_rate=sr, device=self.device)