from flask import request, Blueprint, render_template, flash, redirect, send_file, session, url_for
from .applications.features.speech_translation import preprocess, model
from .applications.features.speech_recognition import recognition
import os
import torch


views = Blueprint('views', __name__)

pretemp = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
TOKEN, MODEL = model.backBone()
TEMP_TXT_FILE = pretemp + '/static/temp.txt'

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


@views.route('/')
def home():
    return render_template("home.html")


@views.route('/translate')
def translate():
    if session.get('downlink'):
        downLink = session.get('downlink')
        input_script = session.get("input")
        output_script = session.get("content")
        session.clear()
        return render_template("translate.html", relink=downLink, input=input_script, content=output_script)
    else:
        return render_template("translate.html")

@views.route('/translate', methods=['GET', 'POST'])
def upload_audio():
    if request.method == "POST":
        if request.form.get('inputNote'):
            text = request.form.get('inputNote')
            session['input'] = text

        else:
            flash('Content conflict', category='error')
            return render_template("translate.html")

    #Infer
    result = model.infer([['test', 'en: ' + text, '']], TOKEN, MODEL, 256, DEVICE)

    #Write result to temp file
    with open(TEMP_TXT_FILE, 'w', encoding="utf-8") as f:
        f.write(result[0][0])

    session['downlink'] = f'/download/{TEMP_TXT_FILE}'
    session['content'] = result[0][0]

    return redirect('/translate')

@views.route('/download/<path:filename>')
def download(filename):
    return send_file(filename, as_attachment=True)
