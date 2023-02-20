import os
import time
import json
import string
from tqdm import tqdm
# import torchaudio
from typing import Union, Any
from pymediainfo import MediaInfo
import warnings
warnings.filterwarnings("ignore")

from .features.speech_enhancement.utils.preprocess import extract_audio_from_video
from .features.speech_enhancement.enhancement import enhance_speech
from .features.speech_recognition.recognition import speech_recognize
from .features.speech_recognition.utils.metrics import cal_average_WER,cal_single_WER


def measure_time_exc_enhance(enhance_file: str, func, **kargs):
    print(f"Measuring {func.__name__} module...")
    st = time.time()
    try:
        assert os.path.exists(enhance_file), f"{enhance_file} not exists"
        fileInfo = MediaInfo.parse(enhance_file)
        for track in fileInfo.tracks:
            if track.track_type == "Video":
                file_type = "video"
            elif track.track_type == "Audio":
                file_type = "audio"
            duration = track.duration
        func(enhance_file, **kargs)
        et = time.time()
        elapses_time = et - st
        print(f"File type: {file_type}")
        print(f"File duration: {duration/1000}s")
    except AssertionError:
        print("[ERROR] File not found")
        elapses_time = 0
        pass

    print(f"Total execution time: {elapses_time}s")


def eval_speech_recog(eval_data_dir: str, gt_value_file: str, lang: str, enh: bool):
    assert os.path.isfile(gt_value_file)
    assert os.path.isdir(eval_data_dir)
    preds = []
    gts = []
    with open(gt_value_file, "r") as f:
        gt_values = json.load(f)
        for file in tqdm(os.listdir(eval_data_dir)):
            pred = speech_recognize(os.path.join(eval_data_dir[:-1], file),
                                    enhance=enh,
                                    lang=lang)["text"].lower().translate(str.maketrans('', '', string.punctuation))
            gt = gt_values[lang][os.path.splitext(file)[0]].lower().translate(str.maketrans('', '', string.punctuation))
            print(f"[EVAL] WER: {cal_single_WER(prediction=[pred], reference=[gt])}")
            preds.append(pred)
            gts.append(gt)

        avg_wer = cal_average_WER(predictions=preds, references=gts)

        print(f"""
                    ####################/n
                    Average WER: {avg_wer}/n
                    ####################/n
                """)


def eval():
    mp3_file = r"/mnt/c/Users/ASUS/Downloads/8191233944245651258.mp4"
    speech_enhanced = enhance_speech(enhance_file=mp3_file)
    speech_recog = speech_recognize(
        speech_enhanced.squeeze(0).cpu().detach().numpy(), False)
    # speech_recog = speech_recognize(mp3_file)

    print(speech_recog)


if __name__ == '__main__':
    eval_speech_recog(eval_data_dir="/content/drive/MyDrive/AI Capstone Project - SP23/data/audio/",
                      gt_value_file="applications/data/captionData.json",
                      lang="en",
                      enh=True)
