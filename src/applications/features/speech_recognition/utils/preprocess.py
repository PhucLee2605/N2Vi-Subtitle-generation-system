import torch
import numpy as np
import librosa as ls


def load_audio_file(batch):
    speech, sr = ls.load(batch["file"])#, mono=False)  # , sr=22050)
    speech = np.expand_dims(speech,axis=0)
    batch["speech"] = torch.from_numpy(speech)
    batch["sampling rate"] = sr
    return batch
