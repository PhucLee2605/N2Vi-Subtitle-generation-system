import torch
import librosa as ls


def map_to_array(batch):
    speech, sr = ls.load(batch["file"], mono=False)  # , sr=22050)
    batch["speech"] = torch.from_numpy(speech)
    batch["sampling rate"] = sr
    return batch
