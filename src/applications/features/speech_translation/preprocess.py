import os
import json
from bs4 import BeautifulSoup


def getCaption(database, vi=True, en=True):
    capPath = os.path.join(database, 'caption')
    captionData = dict()

    if vi:
        viPath = os.path.join(capPath, 'vi')
        viCaptions = os.listdir(viPath)
        viCaptionDict = dict()
        for fileCap in viCaptions:
            try:
                text = extractXml(os.path.join(viPath, fileCap), 'vi')
                viCaptionDict[fileCap[:3]] = text
            except:
                print(fileCap)
        captionData['vi'] = viCaptionDict

    if en:
        enPath = os.path.join(capPath, 'en')
        enCaptions = os.listdir(enPath)
        enCaptionDict = dict()
        for fileCap in enCaptions:
            try:
                text = extractXml(os.path.join(enPath, fileCap), 'en')
                enCaptionDict[fileCap[:3]] = text
            except:
                print(fileCap)
        captionData['en'] = enCaptionDict
    return captionData


def extractXml(data, lang='en'):
    xmlData = BeautifulSoup(data, "xml")
    if lang == 'vi':
        sentence = ' '.join([text.string.replace('\n', ' ') for text in xmlData.find_all('p')[1:]])
    else:
        sentence = ' '.join([text.string.replace('\n', ' ') for text in xmlData.find_all('p')])
    return sentence


def handlejsonData(jsonpath, input='en', output='vi'):
    with open(jsonpath, 'r', encoding="utf-8") as f:
        data = json.load(f)

    results = list()
    for id in data['vi'].keys():
        results.append([id, data[input][id], data[output][id]])

    return results


def splitSentence(sen, threshold):
    out = []
    lang = sen[:4]
    for chunk in sen.split('. '):
        if out and len(chunk) + len(out[-1]) < threshold:
            out[-1] += ' ' + chunk + '.'
        else:
            out.append(lang + chunk + '.')

    return out