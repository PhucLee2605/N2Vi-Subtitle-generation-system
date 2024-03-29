from flask import request, Blueprint, render_template, flash, redirect, send_file, jsonify
from .applications import enhance, recognize, translate
from .applications.utils import preprocess, postprocess, crawl, util
from datetime import datetime
import xml.etree.ElementTree as ET
from hashlib import sha256
import librosa
import os
import torch
import time

start_activate = time.time()

views = Blueprint('views', __name__)

pretemp = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENHANCE_MODEL = enhance.Enhancement(device=DEVICE)
RECOGNIZE_MODEL = recognize.Recognition(device=DEVICE)
TRANSLATE_MODEL = translate.Translation(device=DEVICE)

preprocess.prepare_database()

print(f'[INFO] The system took {int(time.time() - start_activate)}s to start up')

@views.route('/')
def home():
    return render_template("home.html")


@views.route('/translate')
def translate():
    return render_template("translate.html")


@views.route('/translated', methods=['POST'])
def run_translation():
    file_extension = None
    TEMP_XML_DIR = f"{pretemp}/database/translate/xml"
    TEMP_SRT_DIR = f"{pretemp}/database/translate/srt"
    TEMP_TXT_DIR = f"{pretemp}/database/translate/text"

    util.create_dir(os.path.dirname(TEMP_XML_DIR))
    util.create_dir(os.path.dirname(TEMP_SRT_DIR))
    util.create_dir(os.path.dirname(TEMP_TXT_DIR))
    if request.method == "POST":
        print('[INFO] POST')

        if request.form['text']:
            text = request.form['text']
            lang = request.form['lang']
            file_name = request.form['name'].split('.')[0]
            file_extension = request.form['name'].split('.')[-1]

            date = datetime.now().strftime("%Y%m%d%H%M%S")
            file_name = sha256(f"{file_name}{date}".encode()).hexdigest()

            TEMP_TXT_OUTPUT_FILE = f'{TEMP_TXT_DIR}/{file_name}_vi.txt'
        else:
            print("[ERROR] Error translating")
            flash('Content conflict', category='error')
            return redirect("/translate")

    if file_extension == 'xml':
        try:
            xml = ET.fromstring(text)
            TEMP_XML_FILE = f'{TEMP_XML_DIR}/{file_name}_vi.xml'
            TEMP_SRT_FILE = f'{TEMP_SRT_DIR}/{file_name}_vi.srt'
            print("[INFO] XML parse successfully")
        except:
            print("[ERROR] XML parse error")
            flash('XML parse error', category='error')
            return redirect("/translate")

        tags = xml.findall(".//p")
        ptags = [f"{lang}: " + tag.text.strip() for tag in tags]
        ptranslate = [p[4:] for p in TRANSLATE_MODEL.infer(ptags, 'xml')]

        for idex in range(len(ptranslate)):
            tags[idex].text = ptranslate[idex]

        text_data = ' '.join([p.strip() for p in ptranslate])

        with open(TEMP_TXT_OUTPUT_FILE, 'w', encoding="utf-8") as f:
            f.write(text_data)

        with open(TEMP_XML_FILE, "w", encoding="utf-8") as f:
            xml_data = ET.tostring(xml).decode("utf-8")
            f.write(xml_data)
            postprocess.xml_to_srt(xml_data, TEMP_SRT_FILE)

        txtpath = f'/download/{TEMP_TXT_OUTPUT_FILE}'
        xmlpath = f'/download/{TEMP_XML_FILE}'
        srtpath = f'/download/{TEMP_SRT_FILE}'

        return jsonify({"txt_href": txtpath,
                        "xml_href": xmlpath,
                        "srt_href": srtpath})

    elif file_extension == 'srt':
        try:
            srt = preprocess.extract_srt(text)
            TEMP_SRT_FILE = f'{TEMP_SRT_DIR}/{file_name}_vi.srt'
            print("[INFO] SRT parse successfully")
        except:
            print("[ERROR] SRT parse error")
            flash('SRT parse error', category='error')
            return redirect("/translate")

        ptags = [f"{lang}: " + p.strip() for p in srt]
        ptranslate = TRANSLATE_MODEL.infer(ptags)
        text_data = ' '.join([p.strip() for p in ptranslate])

        blocks = text.split('\n\n')

        mod_blocks = list()
        for index in range(len(ptranslate)):
            block = blocks[index]

            lines = block.split('\n')
            new_block = '\n'.join([lines[0], lines[1], ptranslate[index]])
            mod_blocks.append(new_block)

        srt_trans = '\n\n'.join(mod_blocks)

        with open(TEMP_TXT_OUTPUT_FILE, 'w', encoding="utf-8") as f:
            f.write(text_data)

        with open(TEMP_SRT_FILE, "w", encoding="utf-8") as f:
            f.write(srt_trans)

        return jsonify({"txt_href": f'/download/{TEMP_TXT_OUTPUT_FILE}',
                        "srt_href": f'/download/{TEMP_SRT_FILE}'})

    else:
        result = TRANSLATE_MODEL.infer([f'{lang}: ' + text])[0]
        # Write result to temp file
        with open(TEMP_TXT_OUTPUT_FILE, 'w', encoding="utf-8") as f:
            f.write(result)

        return jsonify({"txt_href": f'/download/{TEMP_TXT_OUTPUT_FILE}'})


# RECOGNIZE
@views.route('/recognize')
def recognize():
    return render_template('recognize.html')


@views.route('/recognized', methods=['POST'])
def run_recognition():
    print('[INFO] Recognizing speech')
    if request.method == 'POST':
        start_recog_time = time.time()

        try:
            raw_audio = request.files.get('audio')
        except:
            flash('There is no file provided')
            return redirect('/recognize')

        if raw_audio:
            print('[INFO] File received')
            audio_name = ''.join(raw_audio.filename.split('.')[:-1])

            date = datetime.now().strftime("%Y%m%d%H%M%S")
            audio_name = sha256(f"{audio_name}{date}".encode()).hexdigest()

            audio_temp_dir = f'{pretemp}/database/recognize/audio/temp_{raw_audio.filename}'
            raw_audio.save(audio_temp_dir)
            try:
                speech, sr = preprocess.load_audio_file(audio_temp_dir, 16000)
                audio_duration = librosa.get_duration(filename=audio_temp_dir)
            except:
                return jsonify({"text": "Error: File not supported"}), 400

            enhance_speech, sr = ENHANCE_MODEL.infer(speech, sr)
            transcription = RECOGNIZE_MODEL.infer(enhance_speech, sr)

            if request.form.get('lang') == 'vi':
                lines = ['en: ' + line['text'] for line in transcription['chunks']]
                predict = [p[4:] for p in TRANSLATE_MODEL.infer(lines, 'xml')]

                for i in range(len(predict)):
                    transcription['chunks'][i]['text'] = predict[i]

                text_output = ' '.join(predict)
            else:
                text_output = '.'.join([line['text'] for line in transcription['chunks']]).replace('..', '.')

            txt_audio_output = f"{pretemp}/database/recognize/text/{audio_name}.txt"

            with open(txt_audio_output, 'w', encoding="utf-8") as f:
                f.write(text_output)

            xml_audio_output = f'{pretemp}/database/recognize/xml/{audio_name}.xml'
            srt_audio_output = f'{pretemp}/database/recognize/srt/{audio_name}.srt'


            with open(xml_audio_output, 'w', encoding="utf-8") as f:
                xml_data = postprocess.export_xml(transcription)
                f.write(xml_data)
                postprocess.xml_to_srt(xml_data, srt_audio_output)

            print(f'[INFO] Done audio recognition: audio-duration: {int(audio_duration)}s time-recog: {int(time.time() - start_recog_time)}s')
            return jsonify({"text": text_output,
                            "txt_href": f'/download/{txt_audio_output}',
                            "xml_href": f'/download/{xml_audio_output}',
                            "srt_href": f'/download/{srt_audio_output}'})

        else:
            return redirect('/recognize')
# GET TRANSCRIPT

@views.route('/tubescribe')
def tubescribe():
    return render_template('tubescribe.html')


@views.route('/get_transcribe', methods=['POST'])
def get_transcribe():
    if request.method == 'POST':
        url = request.form['url']
        try:
            audio_name, typp = crawl.download_audio(url, 'src/database/tubescribe')
            srt_output = f'{pretemp}/database/tubescribe/srt/{audio_name}.srt'
        except:
            print('download Fail')
            return jsonify('Error when downloading from provided url'), 404


        print(typp)
        if typp == 'mp4':
            audio_path = os.path.join(f'{pretemp}/database/tubescribe/audio', f'{audio_name}.mp4')

            speech, sr = preprocess.load_audio_file(audio_path, 16000)
            enhance_speech, _ = ENHANCE_MODEL.infer(speech, sr)
            transcription = RECOGNIZE_MODEL.infer(enhance_speech)

            lines = ['en: ' + line['text'] for line in transcription['chunks']]
            predict = TRANSLATE_MODEL.infer(lines, 'xml')

            phrases = postprocess.process_long_text(predict)
            with open(srt_output, 'w', encoding="utf-8") as f:
                for index in range(len(predict)):
                    f.write(f"{index + 1}\n")
                    start_time = transcription['chunks'][index]['timestamp'][0] * 1000
                    end_time = transcription['chunks'][index]['timestamp'][1] * 1000
                    f.write(f"{postprocess.format_time(int(start_time))} --> {postprocess.format_time(int(end_time))}\n")
                    f.write(f"{phrases[index][4:]}\n\n")

        else:
            xml_path = os.path.join(f'{pretemp}/database/tubescribe/xml', f'{audio_name} (en).xml')
            with open(xml_path, 'r') as f:
                try:
                    xml = ET.fromstring(f.read())
                except:
                    raise "Parse XML fail in tubescribe"

            tags = xml.findall(".//p")
            ptags = ["en: " + tag.text.strip() for tag in tags]
            predict = [p[4:] for p in TRANSLATE_MODEL.infer(ptags, 'xml')]

            phrases = postprocess.process_long_text(predict)

            for idex in range(len(phrases)):
              tags[idex].text = phrases[idex]
            
            xml_data = ET.tostring(xml).decode("utf-8")
            postprocess.xml_to_srt(xml_data, srt_output)

        return jsonify({"srt_href": f'/download/{srt_output}'})

    return redirect('/tubescribe')


@views.route('/download/<path:filename>')
def download(filename):
    try:
        return send_file(filename, as_attachment=True)
    #Colab
    except:
        return send_file('/' + filename, as_attachment=True)
