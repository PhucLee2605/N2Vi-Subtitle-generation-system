from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.translate.bleu_score import sentence_bleu
import json
import nltk

def handleData(jsonpath, input='en', output='vi'):
    with open(jsonpath, 'r', encoding="utf-8") as f:
        data = json.load(f)

    results = list()
    for id in data['vi'].keys():
        results.append([id, data[input][id], data[output][id]])

    return results


def bleuScore(sen1, sen2):
  words1 = sen1.split(' ')
  words2 = sen2.split(' ')
  bleuscore = nltk.translate.bleu_score.sentence_bleu([words1], words2)

  return bleuscore


def valuate(data):
  model_name = "VietAI/envit5-translation"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForSeq2SeqLM.from_pretrained(model_name).cuda()

  score = 0
  num = len(data)
  for key, input, output in data:
    tokenInput = tokenizer(input, return_tensors="pt", padding=True).input_ids.to('cuda')
    decodeInput = tokenizer.batch_decode(tokenInput, skip_special_tokens=True)[0]

    sentences = re.sub(' +', ' ', decodeInput.strip()).split('. ')
    label = re.sub(' +', ' ', output.strip()).split('. ')

    if len(sentences) == len(label):
      senScore = 0
      for id in range(len(label)):
        tokenWord = tokenizer("en: " + sentences[id], return_tensors="pt", padding=True).input_ids.to('cuda')
        result = model.generate(tokenWord, max_length=128)
        result = tokenizer.batch_decode(result, skip_special_tokens=True)

        senScore += bleuScore(result[0], label[id])
      score += senScore / len(label)
      print(f"Finish {key}")
    else:
      print(key)
      num -= 1

  return score / num