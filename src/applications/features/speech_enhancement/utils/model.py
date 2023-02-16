from denoiser import pretrained
from typing import Literal, Any
from .util import cfg


def denoise_model(name: Literal["dns48", "dns64", "master64"]) -> Any:
    model = eval("pretrained.{}().to(cfg.getProperty(\"device\"))".format(name))

    return model
