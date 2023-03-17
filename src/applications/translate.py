from typing import Any, Callable, List

from .translation.model import translate_model, infer

#TODO: complete docstring
class Translation():
    def __init__(self, model_name: str = "VietAI/envit5-translation", max_length: int = 256, device: str = 'cpu') -> None:
        """_summary_

        Args:
            model_name (str, optional): _description_. Defaults to "VietAI/envit5-translation".
            max_length (int, optional): _description_. Defaults to 256.
            device (str, optional): _description_. Defaults to 'cpu'.
        """
        self.tokenizer, self.model = translate_model(model_name=model_name,
                                                     device=device)
        self.max_length = max_length
        self.device = device

    def infer(self, texts: List[str], format: str = 'text') -> Callable[[], Any]:
        """_summary_

        Args:
            texts (List[str]): _description_
            format (str, optional): _description_. Defaults to 'text'.

        Returns:
            Callable[[], Any]: _description_
        """
        return infer(texts,
                     self.tokenizer,
                     self.model,
                     self.max_length,
                     format=format,
                     device=self.device)
