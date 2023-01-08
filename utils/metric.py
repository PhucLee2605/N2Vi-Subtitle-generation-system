from nltk.translate.bleu_score import sentence_bleu
from .model import backBone
import re
import torch

def bleuScore(sen1, sen2):
  words1 = sen1.split(' ')
  words2 = sen2.split(' ')
  bleuscore = sentence_bleu([words1], words2)

  return bleuscore


def valuateTranslate(model, data, max_length=128):
  tokenizer, model = backBone(model)

  score = 0
  num = len(data)
  for key, input, output in data:
    if torch.cuda.is_available():
      device = 'cuda'
    else:
      device = 'cpu'

    tokenInput = tokenizer(input, return_tensors="pt", padding=True).input_ids.to(device)
    decodeInput = tokenizer.batch_decode(tokenInput, skip_special_tokens=True)[0]

    sentences = re.sub(' +', ' ', decodeInput.strip()).split('. ')
    label = re.sub(' +', ' ', output.strip()).split('. ')

    if len(sentences) == len(label):
      sen_score = 0
      for id in range(len(label)):
        token_word = tokenizer("en: " + sentences[id], return_tensors="pt", padding=True).input_ids.to(device)
        result = model.generate(token_word, max_length=max_length)
        result = tokenizer.batch_decode(result, skip_special_tokens=True)

        sen_score += bleuScore(result[0], label[id])
      score += sen_score / len(label)
      print(f"Finish {key}")
    else:
      print(key)
      num -= 1

  return score / num

