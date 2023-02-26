import torch
import os
import argparse
import torchaudio
from pymediainfo import MediaInfo
from typing import Any, Union, Literal
from denoiser.dsp import convert_audio
from .utils.util import cfg, model as enhance_model, DEVICE
from .utils.model import denoise_model
from .utils.preprocess import extract_audio_from_video


# TODO: add enhance for already loaded audio
def enhance_speech(audio_file: Union[str, Any], sampling_rate: int = 22500, model: Union[Literal["dns48", "dns64", "master64"], Any] = None) -> Union[Any, int]:
    """This funciton enhance speech

    Args:
        audio_file (str, Any): audio's path or loaded ( to be enhanced
        model (Any): type of pretrained model ("dns48", "dns64", "master64"). Already loaded DNS64, specify if want to use another

    Returns:
        Union[Any, int]: return enhanced speecch in the form of torch.Tensor and its sampling rate 
    """
    if isinstance(model, str):
        model = denoise_model(model)
    else:
        model = enhance_model
    model = model.to(DEVICE)
    try:
        assert os.path.isfile(audio_file)
        fileInfo = MediaInfo.parse(audio_file)
        for track in fileInfo.tracks:
            if track.track_type == "Video":
                enhanced_audio, sr = extract_audio_from_video(audio_file)
            elif track.track_type == "Audio":
                enhanced_audio, sr = torchaudio.load(audio_file)
    except (AssertionError, TypeError):
        enhanced_audio = audio_file
        sr = sampling_rate
        
    enhanced_audio = convert_audio(wav=enhanced_audio.to(DEVICE), 
                                   from_samplerate=sr, 
                                   to_samplerate=model.sample_rate, 
                                   channels=model.chin)

    with torch.no_grad():
        denoised = model(enhanced_audio[None])[0]
        denoised = torch.mean(denoised, dim=0, keepdim=True)

    print("[INFO] Finished speech enhancement")
    return denoised, sr


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
