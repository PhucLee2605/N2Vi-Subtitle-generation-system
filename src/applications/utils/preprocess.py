import os
import torch
import json
import numpy as np
import librosa as ls
import xml.etree.ElementTree as ET
from moviepy.editor import VideoFileClip
import torchaudio
from pymediainfo import MediaInfo
#Enhance and Recognition


def load_audio_file(audio):
    try:
        assert os.path.isfile(audio)
        fileInfo = MediaInfo.parse(audio)
        for track in fileInfo.tracks:
            if track.track_type == "Video":
                speech, sr = extract_audio_from_video(audio)
            elif track.track_type == "Audio":
                speech, sr = ls.load(audio)
    except:
        speech, sr = ls.load(audio)

    speech = np.expand_dims(speech, axis=0)

    enhanced_audio = torch.from_numpy(speech)
    return enhanced_audio, sr


def extract_audio_from_video(video_file: str):
    """_summary_

    Args:
        video_file (str): _description_

    Returns:
        _type_: _description_
    """
    print("[INFO] Video file passed in. Extracting audio from video...")
    video = VideoFileClip(video_file)
    audio = video.audio.to_soundarray().T
    sr = video.audio.fps
    return audio, sr



#Translate
def get_caption(database, vi=True, en=True):
    '''

    :param database: Path of database
    :param vi: Vietnamese or not
    :param en: English or not
    :return: extract transcript data from database

    '''
    capPath = os.path.join(database, 'caption')
    captionData = dict()

    if vi:
        viPath = os.path.join(capPath, 'vi')
        viCaptions = os.listdir(viPath)
        viCaptionDict = dict()
        for fileCap in viCaptions:
            try:
                text = extract_xml(os.path.join(viPath, fileCap), 'vi')
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
                text = extract_xml(os.path.join(enPath, fileCap), 'en')
                enCaptionDict[fileCap[:3]] = text
            except:
                print(fileCap)
        captionData['en'] = enCaptionDict
    return captionData


def extract_xml(path, lang='en'):
    '''
    Extract information from xml transcript files pull from Youtube channel Ted-ed

    :param path: Path to XML file
    :param lang: Transcript language Vietnamese or English
    :return: A paragraph by attaching transcript lines

    '''
    with open(path, 'r', encoding="utf-8") as f:
        xmlData = ET.fromstring(f.read())

    if lang == 'vi':
        sentence = ' '.join([p.text for p in xmlData.findall('.//p')[1:]])
    else:
        sentence = ' '.join([p.text for p in xmlData.findall('.//p')])
    return sentence


def handle_json_data(jsonpath, input='en', output='vi'):
    '''

    Read json file extract from get_caption function

    '''
    with open(jsonpath, 'r', encoding="utf-8") as f:
        data = json.load(f)

    results = list()
    for id in data['vi'].keys():
        results.append([id, data[input][id], data[output][id]])

    return results