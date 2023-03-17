from pytube import YouTube
import os
import json
import urllib
from typing import Any
from hashlib import sha256


#TODO complete docstring
def download_audio(url: str, path: str) -> Any:
    """_summary_

    Args:
        url (str): _description_
        path (str): _description_

    Returns:
        Any: _description_
    """
    yt = YouTube(url)

    t = yt.streams.filter(only_audio=True)

    name_encrypted = sha256(yt.streams[0].title.encode('utf-8')).hexdigest()
    t[0].download(filename=f"{name_encrypted}.mp4", output_path=path)
    return name_encrypted


#TODO complete docstring and type hint
def crawl_data(channelid, api, num) -> None:
    """_summary_

    Args:
        channelid (_type_): _description_
        api (_type_): _description_
        num (_type_): _description_
    """
    nextPage = None
    filename = 0
    while True:
        if 0 < num < 50:
            if nextPage == None:
                url = f"https://www.googleapis.com/youtube/v3/search?order=date&part=snippet&channelId={channelid}&maxResults={num}&key={api}"
            else:
                url = f"https://www.googleapis.com/youtube/v3/search?order=date&part=snippet&pageToken={nextPage}&channelId={channelid}&maxResults={num}&key={api}"
            googleResponse = urllib.request.urlopen(url)
            youtubeResponse = json.loads(googleResponse.read())

        elif num >= 50:
            if nextPage == None:
                url = f"https://www.googleapis.com/youtube/v3/search?order=date&part=snippet&channelId={channelid}&maxResults=50&key={api}"
            else:
                url = f"https://www.googleapis.com/youtube/v3/search?order=date&part=snippet&pageToken={nextPage}&channelId={channelid}&maxResults=50&key={api}"
            googleResponse = urllib.request.urlopen(url)
            youtubeResponse = json.loads(googleResponse.read())
            nextPage = youtubeResponse['nextPageToken']

            num -= 50

        else:
            return

        for item in youtubeResponse['items']:
            id = item['id']['videoId']
            vidPath = f"https://www.youtube.com/watch?v={id}"
            extract_info(str(filename).zfill(3), vidPath)

            filename += 1


#TODO complete docstring and type hint
def extract_info(filename, link, database="data") -> None:
    """_summary_

    Args:
        filename (_type_): _description_
        link (_type_): _description_
        database (str, optional): _description_. Defaults to "data".
    """
    audio_path = os.path.join(database, "audio")
    vi_cap_path = os.path.join(database, "caption/vi")
    en_cap_path = os.path.join(database, "caption/en")

    if not os.path.isdir(database) or not os.path.exists(database):
        os.makedirs(audio_path)
        os.makedirs(vi_cap_path)
        os.makedirs(en_cap_path)

    yt = YouTube(link)
    try:
        vi_cap = yt.captions['vi']
    except:
        vi_cap = None
    try:
        en_cap = yt.captions['en']
    except:
        en_cap = None

    audio = yt.streams.filter(only_audio=True, file_extension='mp3').first()

    0 if vi_cap == None else vi_cap.download(filename,
                                             srt=False,
                                             output_path=vi_cap_path)

    en_cap.download(filename, srt=False, output_path=en_cap_path)
    audio.download(output_path=audio_path, filename=f'{filename}.mp3')
    print(f'[INFO] Load {filename} succesfully')
