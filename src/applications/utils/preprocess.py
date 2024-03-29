import os
from typing import Any, Dict, Tuple, Union
import torch
import json
import numpy as np
import librosa as ls
import xml.etree.ElementTree as ET
from moviepy.editor import VideoFileClip
import re
from pymediainfo import MediaInfo


def load_audio_file(audio: str, output_sr: int = 16000) -> Tuple[Any, int]:
    """ Load audio from either video or audio file

    Args:
        audio (str): file path to be loaded
        output_sr (int, optional): Expected sampling rate that audio will be loaded (in Hz). Defaults to 16000.

    Returns:
        Tuple[Any, int]: speech loaded from audio file or extracted from video file and sampling rate
    """
    assert os.path.isfile(audio)
    fileInfo = MediaInfo.parse(audio)
    speech = None

    for track in fileInfo.tracks:
        if track.track_type == "Video":
            speech, sr = extract_audio_from_video(audio)
        elif track.track_type == "Audio":
            speech, sr = ls.load(audio, sr=output_sr)

    #! check if audio is loaded. If not, means that input file is not supported
    if speech is None:
        print("[ERROR] File is not supproted")
        raise

    speech = np.expand_dims(speech, axis=0)

    speech = torch.from_numpy(speech)
    return (speech, sr)


def extract_audio_from_video(video_file: str) -> Tuple[Any, int]:
    """ Extract audio from video input file, return extracted speech in form of torch tensor and its sampling rate

    Args:
        video_file (str): video path to extract

    Returns:
        Tuple[Any, int]: extracted speech and its sampling rate
    """
    print("[INFO] Video file passed in. Extracting audio from video...")
    video = VideoFileClip(video_file)
    audio = video.audio.to_soundarray().T
    audio = torch.from_numpy(audio)
    sr = video.audio.fps
    print("[INFO] Extract audio successfully")
    return audio, sr


#TODO complete docstring
def get_caption(database, vi=True, en=True) -> Dict:
    """_summary_

    Args:
        database (_type_): _description_
        vi (bool, optional): _description_. Defaults to True.
        en (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
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


# TODO complete docstringg
def extract_xml(path, lang='en') -> str:
    """_summary_

    Args:
        path (_type_): _description_
        lang (str, optional): _description_. Defaults to 'en'.

    Returns:
        str: _description_
    """
    with open(path, 'r', encoding="utf-8") as f:
        xmlData = ET.fromstring(f.read())

    if lang == 'vi':
        sentence = ' '.join([p.text for p in xmlData.findall('.//p')[1:]])
    else:
        sentence = ' '.join([p.text for p in xmlData.findall('.//p')])
    return sentence


# TODO: complete docstring
def handle_json_data(jsonpath, input='en', output='vi'):
    """_summary_

    Args:
        jsonpath (_type_): _description_
        input (str, optional): _description_. Defaults to 'en'.
        output (str, optional): _description_. Defaults to 'vi'.

    Returns:
        _type_: _description_
    """
    english_caption = list()
    vietnamese_caption = list()

    with open(jsonpath, 'r', encoding='utf-8') as f:
      data = json.load(f)

      for id in data['vi'].keys():
        try:
          english_caption.append(data['en'][id].replace('\n', ' '))
          vietnamese_caption.append(data['vi'][id].replace('\n', ' '))
        except:
          print('Error: ' + id)

    return english_caption, vietnamese_caption


def is_srt_format(string: str) -> bool:
    """ check if input string is in srt format or not

    Args:
        string (str): input string to check if it is in srt format 

    Returns:
        bool: True if input string is in srt and False if it is not
    """
    blocks = string.strip().split('\n\n')
    for block in blocks:
        lines = block.split('\n')

        if 3 > len(lines) > 5:
            print(len(lines))
            return False

        if not lines[0].isdigit():
            return False
        timecode = lines[1].split(' --> ')
        if len(timecode) != 2:
            return False
        for tc in timecode:
            if not re.match(r'\d{2}:\d{2}:\d{2},\d{3}', tc):
                return False
    return True


# TODO complete docstring
def extract_srt(str: str):
    """_summary_

    Args:
        str (_type_): _description_

    Raises:
        ValueError: raise if input string is not in srt format

    Returns:
        _type_: _description_
    """
    if not is_srt_format(str):
        raise ValueError('Not a valid SRT file')
    blocks = str.strip().split('\n\n')
    results = list()
    for block in blocks:
        lines = block.split('\n')
        results.append('\n'.join(lines[2:]))
    return results


def gaussian_noise(audio, noiseSigma = 0.1, noiseAmplitude = 5):
    Nsamples = len(audio[0])
    noise = noiseAmplitude * np.random.normal(0, noiseSigma, Nsamples)
    
    audio_noise = audio[0] + torch.Tensor(noise)
    audio_noise = audio_noise.reshape(1, len(audio_noise))
    return audio_noise

def prepare_database():
    os.makedirs('src/database/recognize/audio', exist_ok=True)
    os.makedirs('src/database/recognize/srt', exist_ok=True)
    os.makedirs('src/database/recognize/text', exist_ok=True)
    os.makedirs('src/database/recognize/xml', exist_ok=True)

    #translate
    os.makedirs('src/database/translate/srt', exist_ok=True)
    os.makedirs('src/database/translate/xml', exist_ok=True)
    os.makedirs('src/database/translate/text', exist_ok=True)

    #tubescribe
    os.makedirs('src/database/tubescribe/srt', exist_ok=True)
    os.makedirs('src/database/tubescribe/audio', exist_ok=True)
    os.makedirs('src/database/tubescribe/xml', exist_ok=True)
    print('[INFO] Database preparation successfully')
