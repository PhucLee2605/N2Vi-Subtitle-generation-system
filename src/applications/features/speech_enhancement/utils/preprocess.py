from moviepy.editor import VideoFileClip


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
