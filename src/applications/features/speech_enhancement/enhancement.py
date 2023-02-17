import torch
import os
import argparse
import torchaudio
from typing import Any, Union, Literal
from denoiser.dsp import convert_audio
from .utils.util import cfg
from .utils.model import denoise_model

#TODO: add enhance for already loaded audio
def enhance_speech(audio_file: str, model: Literal["dns48", "dns64", "master64"]) -> Union[Any, int]:
    """This funciton enhance speech

    Args:
        audio_file (str): audio's path to be enhanced
        model (Any): type of pretrained model ("dns48", "dns64", "master64")

    Returns:
        Union[Any, int]: return enhanced speecch in the form of torch.Tensor and its sampling rate 
    """
    assert os.path.exists(audio_file), f"{audio_file} not exists"
    model = denoise_model(model)
    enhanced_audio, sr = torchaudio.load(audio_file)
    enhanced_audio = convert_audio(
        enhanced_audio.to(), sr, model.sample_rate, model.chin)

    with torch.no_grad():
        denoised = model(enhanced_audio[None])[0]
        denoised = torch.mean(denoised, dim=0, keepdim=True)
    print("[INFO] Finished speech enhancement")
    return denoised


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True,
                        help="file for enhancement")
    parser.add_argument("-dst", "--destination", required=True,
                        help="save destination")
    args = parser.parse_args()

    model_name = cfg.getProperty("model")
    model = denoise_model(model_name)
    enhanced_audio, sr = enhance_speech(args.file, model)
    enhanced_audio = enhanced_audio / max(enhanced_audio.abs().max().item(), 1)

    torchaudio.save(args.destination,
                    enhanced_audio.cpu(),
                    cfg.getProperty("sampling_rate"))
