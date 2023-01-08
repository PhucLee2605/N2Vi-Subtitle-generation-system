from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
import torch

def backBone(name):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    if name == "envit5":
        model_name = "VietAI/envit5-translation"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

        return tokenizer, model

    elif name == "phobert":
        model_name = "vinai/phobert-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)

        return tokenizer, model