from .features.speech_enhancement.enhancement import enhance_speech
from .features.speech_recognition.recognition import speech_recognize


def eval():
    mp3_file = r"/mnt/c/Users/ASUS/Downloads/013c-splitteed.mp3"
    speech_enhanced = enhance_speech(audio_file=mp3_file, model="dns64")
    speech_recog = speech_recognize(speech_enhanced.squeeze(0).cpu().detach().numpy())
    # speech_recog = speech_recognize(mp3_file)

    print(speech_recog)


if __name__ == '__main__':
    eval()
