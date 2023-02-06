from utils.model import enhance_pipeline
from utils.preprocess import map_to_array
import torch
import config_with_yaml as config


cfg = config.load(r"utils/config.yaml")
chunk_length = cfg.getProperty("chunk_lenght_s")
ds = map_to_array({
    "file": r"/mnt/c/Users/ASUS/Downloads/013.mp3"
})


with torch.no_grad():
    transcription = enhance_pipeline()(ds["speech"], chunk_length_s=chunk_length)["text"]

print(transcription)