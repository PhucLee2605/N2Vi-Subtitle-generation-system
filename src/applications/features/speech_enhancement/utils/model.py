from denoiser import pretrained
from typing import Literal, Any


def denoise_model(name: Literal["dns48", "dns64", "master64"]) -> Any:
    model = eval("pretrained.{}()".format(name))

    return model
