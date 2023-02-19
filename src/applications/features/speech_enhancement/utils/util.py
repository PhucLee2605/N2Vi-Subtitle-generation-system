import config_with_yaml as config
import os
import torch
from .model import denoise_model


cfg = config.load(os.path.join(os.path.dirname(__file__), "config.yaml"))

if cfg.getProperty("device"):
    DEVICE = cfg.getProperty("device")
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = denoise_model(cfg.getProperty("model"))
