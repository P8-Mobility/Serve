import configparser
import json
import os
from os import path
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
import classifier
import processing.audio as audio
from language import Language

config_file = "config.cfg"
app = Flask(__name__)


@app.route('/models', methods=['GET'])
def models():
    the_models = []

    for dirname in os.listdir("models/"):
        if dirname != ".gitkeep":
            the_models.append(dirname)

    return jsonify({'status': 'OK', 'result': the_models})


@app.route('/words', methods=['GET'])
def words():
    words = lang.words()
    return jsonify({'status': 'OK',  'result': [{key: value} for key, value in words.items()]})


@app.route('/predict', methods=['POST'])
def predict():
    # 1) Fetch and prepare file
    if 'file' not in request.files:
        return jsonify({'status': 'FAILED', 'message': 'No audio file attached'})

    file = request.files['file']

    # Args
    args = request.form
    model = args.get('model')
    word = args.get('word')

    if model is None:
        model = ""

    if file.filename == '':
        return jsonify({'status': 'FAILED', 'message': 'No file selected'})

    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    filepath = app.config['UPLOAD_FOLDER'] + "/" + filename

    # 2) Load file as wav
    if not os.path.isfile(filepath):
        return jsonify({'status': 'FAILED', 'message': 'File was not saved...'})

    # Check file type
    if filename.endswith(".mp4"):
        filepath = convert_file(filepath)

    audio_file = audio.load(filepath)

    # 3) Preprocessering
    # transformer.remove_noise(audio_file)
    # transformer.trim(audio_file, 25)

    # 4) Predict
    prediction, model = classifier.predict_word(audio_file, model)

    # % Prepare feedback
    dan, ara = classifier.prepare_feedback(word, prediction)

    # 5) Housekeeping
    if os.path.exists(filepath):
        os.remove(filepath)

    # 6) Return success or error depending on prediction
    return jsonify({'status': 'OK', 'phonemes': prediction, 'model': model, 'response_target_lang': dan, 'response_native_lang': ara, 'word': word})


def allowed_file_type(filename, allowed_type):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == allowed_type


def convert_file(filepath):
    new_file = filepath[:len(filepath) - 4]
    os.system('ffmpeg -i {} -vn -acodec pcm_s16le -ar 44100 {}.wav'.format(filepath, new_file))
    os.remove(filepath)
    return new_file + ".wav"


if __name__ == '__main__':
    if not path.exists(config_file) or not path.isfile(config_file):
        print("Config file missing... (config.cfg)")

    config = configparser.ConfigParser()
    config.read(config_file)

    lang = Language()
    lang.load("words.json")

    # Setup upload folder
    app.config['UPLOAD_FOLDER'] = config['UPLOAD']['folder']

    classifier = classifier.Classifier(config, lang)
    if classifier.load_models() and config.getboolean("WEB", "local"):
        app.run(port=int(config['WEB']['port']), host=str(config['WEB']['host']))
