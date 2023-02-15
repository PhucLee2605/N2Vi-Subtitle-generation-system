from flask import request, Blueprint, render_template, flash, redirect, send_file, jsonify
from .applications.features.speech_translation import preprocess, model
from .applications.features.speech_recognition import recognition
import xml.etree.ElementTree as ET
import os
import torch


views = Blueprint('views', __name__)

pretemp = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
TOKEN, MODEL = model.backBone()

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


@views.route('/')
def home():
    return render_template("home.html")


@views.route('/translate')
def translate():
    return render_template("translate.html")

@views.route('/translated', methods=['POST'])
def run_translate():
    if request.method == "POST":
        print('POST')
        if request.form['text']:
            text = request.form['text']
            file_name = request.form['name'].split('.')[0]

            TEMP_TXT_OUTPUT_FILE = f'{pretemp}/static/{file_name}_vi.txt'
            TEMP_XML_FILE = f'{pretemp}/static/{file_name}_vi.xml'
            TEMP_SRT_FILE = f'{pretemp}/static/{file_name}_vi.srt'

            xml = None
            try:
                xml = ET.fromstring(text)
                print("XML parse success")
            except:
                print("xml fail")
        else:
            print("Error")
            flash('Content conflict', category='error')
            return redirect("/translate")

    if xml:
        tags = xml.findall(".//p")
        ptags = ["en: " + tag.text.strip() for tag in tags]
        ptranslate = [p[4:] for p in model.infer(ptags, TOKEN, MODEL, 256, 'xml', DEVICE)]

        for idex in range(len(ptranslate)):
            tags[idex].text = ptranslate[idex]

        text_data = ' '.join([p.strip() for p in ptranslate])

        with open(TEMP_TXT_OUTPUT_FILE, 'w', encoding="utf-8") as f:
            f.write(text_data)

        with open(TEMP_XML_FILE, "w", encoding="utf-8") as f:
            xml_data = ET.tostring(xml).decode("utf8")
            f.write(xml_data)
            preprocess.xml_to_srt(xml_data, TEMP_SRT_FILE)

        txtpath = f'/download/{TEMP_TXT_OUTPUT_FILE}'
        xmlpath = f'/download/{TEMP_XML_FILE}'
        srtpath = f'/download/{TEMP_SRT_FILE}'

        return jsonify({"text": text_data,
                        "txt_href": txtpath,
                        "xml_href": xmlpath,
                        "srt_href": srtpath})


    else:
        result = model.infer(['en: ' + text], TOKEN, MODEL, 256, DEVICE)[0][4:]
        #Write result to temp file
        with open(TEMP_TXT_OUTPUT_FILE, 'w', encoding="utf-8") as f:
            f.write(result)

        return jsonify({"text": result,
                        "txt_href": f'/download/{TEMP_TXT_OUTPUT_FILE}'})


@views.route('/download/<path:filename>')
def download(filename):
    return send_file(filename, as_attachment=True)
