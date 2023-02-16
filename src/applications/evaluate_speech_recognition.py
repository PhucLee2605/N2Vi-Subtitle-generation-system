'''
    If captions of audios aregathered, the caption-gathered file must in the form of:
    {
        "vi":{
            "audio1.mp3": "phụ đề của audio 1",
            "audio2.mp3": "phụ đề của audio 2",
            ...
        },
        "en":{
            "audio1.mp3": "audio 1 caption",
            "audio2.mp3": "audio 2 caption",
            ...
        }
    }
'''

import config_with_yaml as config
import json
import os
from utils.util import cfg
from recognition import speech_recognize
from utils.metrics import cal_average_WER

# TODO: test evaluate
audio_path = cfg.getProperty("audio_path")
caption_path = cfg.getProperty("caption_path")

caption_json = cfg.getProperty("caption_json")

if caption_json:
    with open(caption_path) as caption_file:
        en_captions = json.loads(caption_file.read())["en"]

pred_subs = []
ref_subs = []
for audio in os.listdir(audio_path):
    pred_subs.append(speech_recognize(os.path.join(caption_path, audio)))
    ref_subs.append(en_captions[audio])

print(f"""
        ####################\n
        Average WER: {cal_average_WER(pred_subs,ref_subs)}\n
        ####################\n
      """)
