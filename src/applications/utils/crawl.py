from pytube import YouTube
import os
import json
import urllib
from hashlib import sha256


def download_audio(url, path):
    yt = YouTube(url)

    t = yt.streams.filter(only_audio=True)

    name_encrypted = sha256(yt.streams[0].title.encode('utf-8')).hexdigest()
    t[0].download(filename=f"{name_encrypted}.mp4", output_path=path)
    return name_encrypted


def crawlData(channelid, api, num):
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
            extractInfo(str(filename).zfill(3), vidPath)

            filename += 1


def extractInfo(filename, link, database="data"):
    audioPath = os.path.join(database, "audio")
    viCapPath = os.path.join(database, "caption/vi")
    enCapPath = os.path.join(database, "caption/en")

    if not os.path.isdir(database) or not os.path.exists(database):
        os.makedirs(audioPath)
        os.makedirs(viCapPath)
        os.makedirs(enCapPath)

    yt = YouTube(link)
    try:
        viCap = yt.captions['vi']
    except:
        viCap = None
    try:
        enCap = yt.captions['en']
    except:
        enCap = None

    audio = yt.streams.filter(only_audio=True, file_extension='mp3').first()

    0 if viCap == None else viCap.download(filename, srt=False, output_path=viCapPath)

    enCap.download(filename, srt=False, output_path=enCapPath)
    audio.download(output_path=audioPath, filename=f'{filename}.mp3')
    print(f'Load {filename} succesfully')
