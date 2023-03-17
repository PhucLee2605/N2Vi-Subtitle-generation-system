from denoiser import pretrained
from typing import Literal, Any


def denoise_model(name: Literal["dns48", "dns64", "master64"]) -> Any:
    """ Create model for speech enhancement and minimize background noise

    Args:
        name (Literal['dns48', 'dns64', 'master64']): pretrained model's name

    Returns:
        Any: enhance model
    """
    model = eval("pretrained.{}()".format(name))
    return model
