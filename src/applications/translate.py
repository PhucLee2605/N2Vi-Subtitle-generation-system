from .translation.model import translate_model, infer


class Translation():
    def __init__(self, model_name="VietAI/envit5-translation", max_length=256, device='cpu'):
        self.tokenizer, self.model = translate_model(model_name=model_name, device=device)
        self.max_length = max_length
        self.device = device

    def infer(self, texts, format='text'):
        return infer(texts, self.tokenizer, self.model, self.max_length, format=format, device=self.device)