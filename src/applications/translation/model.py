from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Any, List, Tuple
import re

#TODO: complete docstring
def translate_model(model_name: str, device: str) -> Tuple[Any, Any]:
    """_summary_

    Args:
        model_name (str): _description_
        device (str): _description_

    Returns:
        Tuple[Any, Any]: _description_
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    print("[INFO] Model loaded successfully")
    return (tokenizer, model)

#TODO: complete docstring
def split_sentence(sen: str, threshold: str) -> List[str]:
    """_summary_

    Args:
        sen (str): _description_
        threshold (str): _description_

    Returns:
        List[str]: _description_
    """
    out = []
    lang = sen[:4]
    for chunk in sen.split('. '):
        if out and len(chunk) + len(out[-1]) < threshold:
            out[-1] += ' ' + chunk + '.'
        else:
            out.append(lang + chunk + '.')

    return out

#TODO: complete docstring
def infer(texts: List[str], tokenizer, model, max_length, format: str = 'text', device: str = 'cpu') -> List[str]:
    """_summary_

    Args:
        texts (List[str]): _description_
        tokenizer (_type_): _description_
        model (_type_): _description_
        max_length (_type_): _description_
        format (str, optional): _description_. Defaults to 'text'.
        device (str, optional): _description_. Defaults to 'cpu'.

    Returns:
        List[str]: _description_
    """
    predictions = list()

    if format == 'text':
        for text in texts:
            print(f"[INFO] Starting prediction")
            document = list()
            paragraphs = re.sub(r'(\n\s*){2,}', r'\n\n', text).split('\n')

            for para in paragraphs:
                if para == "":
                    document.append("")
                    continue
                if para[:3] != 'en:' and para[:3] != 'vi:':
                    para = paragraphs[0][:3] + ' ' + para

                token_input = tokenizer(para,
                                        return_tensors="pt",
                                        padding=True).input_ids.to(device)
                decode_input = tokenizer.batch_decode(token_input,
                                                      skip_special_tokens=True)[0]

                sentences = re.sub(' +', ' ', decode_input.strip())

                lstSen = split_sentence(sentences, 256)
                tok = tokenizer(lstSen,
                                return_tensors="pt",
                                padding=True).input_ids.to(device)
                result = model.generate(tok, max_length=max_length)
                result = tokenizer.batch_decode(result,
                                                skip_special_tokens=True)

                pred = ' '.join([chunk[4:] for chunk in result])

                document.append(pred)

            predictions.append('\n'.join(document))

    else:
        tok = tokenizer(texts,
                        return_tensors="pt",
                        padding=True).input_ids.to(device)
        result = model.generate(tok,
                                max_length=max_length,
                                max_time=10.,
                                no_repeat_ngram_size=5)
        predictions = tokenizer.batch_decode(result,
                                             skip_special_tokens=True)

    print('[INFO] Done translation')
    return predictions
