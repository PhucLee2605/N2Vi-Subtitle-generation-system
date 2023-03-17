import os
import time
import json
import string
from typing import Callable
from tqdm import tqdm
from pymediainfo import MediaInfo
import warnings
import recognize
import enhance

from .utils import preprocess
from .recognition.metric import cal_wer_score

warnings.filterwarnings("ignore")

def measure_time_exc(enhance_file: str, func: Callable, **kargs) -> None:
    """ Used to measure time execution of passed function.

    Args:
        enhance_file (str): file used to measure time execution.
        func (Callable): Function that will be measured
    """
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


def eval_speech_recog(eval_data_dir: str, gt_value_file: str, lang: str, enh: bool) -> None:
    """ Used to evaluate speech recognition model

    Args:
        eval_data_dir (str): directory that contain files that will be used to evaluate
        gt_value_file (str): true subtitle file
        lang (str): "en" for english evaluation. "vi" for Vietnamese evaluation
        enh (bool): either enhance speech before evaluate or not
    """    
    assert os.path.isfile(gt_value_file)
    assert os.path.isdir(eval_data_dir)
    enhance_model = enhance.Enhancement()
    recognize_model = recognize.Recognition()
    preds = []
    gts = []
    with open(gt_value_file, "r") as f:
        gt_values = json.load(f)
        for file in tqdm(os.listdir(eval_data_dir)):
            speech, sr = preprocess.load_audio_file(os.path.join(eval_data_dir[:-1], file),16000)
            enhance_audio, _ = enhance_model(speech, sr)

            pred = recognize_model.infer(enhance_audio)["text"].lower().translate(str.maketrans('', '', string.punctuation))
            gt = gt_values[lang][os.path.splitext(file)[0]].lower().translate(str.maketrans('', '', string.punctuation))
            print(f"[EVAL] WER: {cal_wer_score(prediction=[pred], reference=[gt])}")
            preds.append(pred)
            gts.append(gt)

        avg_wer = cal_wer_score(predictions=preds, references=gts)

        print(f"""
                    ####################/n
                    Average WER: {avg_wer}/n
                    ####################/n
                """)


def eval():
    mp3_file = r"/mnt/c/Users/ASUS/Downloads/8191233944245651258.mp4"
    # speech_enhanced = enhance_speech(enhance_file=mp3_file)
    # speech_recog = speech_recognize(
    #     speech_enhanced.squeeze(0).cpu().detach().numpy(), False)
    # root = "/mnt/c/Users/ASUS/OneDrive/Documents/Capstone project sp23/audio"
    # for mp3_file in ["000.mp3"]:#,'006.mp4','002.mp4']:
    # ds = load_audio_file({
    #     # "file": os.path.join(root, mp3_file)
    #     'file': mp3_file
    # })
    # print(ds['speech'].size)
    # speech_recog = enhance_speech(ds["speech"],ds["sampling rate"])#, enhance=False)
    #
    # print(speech_recog)


if __name__ == '__main__':
    # eval_speech_recog(eval_data_dir="/mnt/c/Users/ASUS/OneDrive/Documents/Capstone project sp23/audio/",
    #                   gt_value_file="applications/data/captionData.json",
    #                   lang="en",
    #                   enh=True)
    eval()
