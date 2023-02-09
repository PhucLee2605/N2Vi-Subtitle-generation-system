from flask import request, Blueprint, render_template, flash, redirect, send_file, session, url_for
from .applications.features.speech_translation import preprocess, model
from .applications.features.speech_recognition import recognition
import os
import torch


views = Blueprint('views', __name__)

pretemp = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
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
    try:
        print('ok')
        return render_template("translate.html", relink=session.get('downlink'), input=session.get("input"), content=session.get("content"))
    except:
        return render_template("translate.html", relink=session.get('downlink'))

@views.route('/translate', methods=['POST'])
def upload_audio():
    if request.method == "POST":
        if request.files['file-script'].content_length > 0:
            file = request.files['file-script']

            if file.filename.endswith('.txt'):
                text = file.read().decode("utf-8")
            elif file.filename.endswith('.xml'):
                text = preprocess.extractXml(file.read().decode("utf-8"))
            else:
                flash('No file selected for uploading')
                return(redirect(request.url))

        elif request.files['file-script'].content_length == 0 and request.form.get('inputNote'):
            text = request.form.get('inputNote')
            NOTE = True

        else:
            flash('Content conflict', category='error')
            return render_template("translate.html")

    #Infer
    tok, mod = model.backBone()
    result = model.infer([['test', 'en: ' + text, '']], tok, mod, 256, DEVICE)

    #Write result to temp file
    with open(TEMP_TXT_FILE, 'w', encoding="utf-8") as f:
        f.write(result[0][0])

    session['downlink'] = f'/download/{TEMP_TXT_FILE}'
    if NOTE:
        session['input'] = text
        session['content'] = result[0][0]
    return redirect("/translate")

@views.route('/download/<path:filename>')
def download(filename):
    return send_file(filename, as_attachment=True)


@views.route('/recognition')
def recognize():
    recognition()
    return "<p>Hello</p>"