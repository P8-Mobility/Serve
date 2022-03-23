import configparser
import os
from os import path
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
import classifier
from processing.audio import Audio
import processing.audio as audio
import processing.transformer as transformer
import pre_processing

config_file = "config.cfg"
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():

    # 1) Fetch and prepare file
    if 'file' not in request.files:
        return jsonify({'status': 'FAILED', 'message': 'No audio file attached'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'status': 'FAILED', 'message': 'No file selected'})

    if not file or not allowed_file_type(file.filename, config['UPLOAD']['allowed_type']):
        return jsonify({'status': 'FAILED', 'message': 'File type is not allowed'})

    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    filepath = app.config['UPLOAD_FOLDER'] + "/" + filename

    # 2) Load file as wav
    if not os.path.isfile(filepath):
        return jsonify({'status': 'FAILED', 'message': 'File was not saved...'})

    audio_file = audio.load(filepath)

    # 3) Preprocessering
    transformer.remove_noise(audio_file)
    transformer.normalize(audio_file)
    transformer.trim(audio_file, 20)
    audio_file.save(filepath)

    # 4) Predict
    result = classifier.predict_word(audio_file)

    # 5) Clean up
    if os.path.exists(filepath):
        os.remove(filepath)

    # 6) Return success or error depending on prediction
    return jsonify({'status': 'OK', 'message': 'File saved!'})


def allowed_file_type(filename, allowed_type):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == allowed_type


if __name__ == '__main__':
    if not path.exists(config_file) or not path.isfile(config_file):
        print("Config file missing... (config.cfg)")

    config = configparser.ConfigParser()
    config.read(config_file)

    # Setup upload folder
    app.config['UPLOAD_FOLDER'] = config['UPLOAD']['folder']

    classifier = classifier.Classifier(config)
    if classifier.load_models():
        app.run(port=int(config['WEB']['port']), host=str(config['WEB']['host']))



