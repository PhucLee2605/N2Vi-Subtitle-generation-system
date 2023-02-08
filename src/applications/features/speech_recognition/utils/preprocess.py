import librosa as ls


def map_to_array(batch):
    speech, sr = ls.load(batch["file"])  # , sr=22050)
    batch["speech"] = speech
    batch["sampling rate"] = sr
    return batch
