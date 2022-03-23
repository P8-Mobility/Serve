import configparser
from os import path
from flask import Flask, jsonify, request
import classifier
import helper
import pre_processing

config_file = "config.cfg"
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    json_ = request.json

    if not isinstance(json_, list):
        return jsonify({'status': 'FAILED', 'message': 'Input must be of type list'})

    # Merge rows
    rows = pre_proc.merged_rows(json_, helper.sensor_tags)

    # Do exercise prediction
    exercise, exercise_confidence, exercise_message = classifier.predict_exercise(rows.copy())

    if exercise == "Unknown":
        return jsonify({'status': 'FAILED', 'message': exercise_message})

    if exercise == "00000000-0000-0000-0000-000000000000":
        return jsonify({'status': 'OK',
                        'exercise': str(exercise),
                        'exercise_score': exercise_confidence,
                        'mistakes': 0,
                        'mistakes_score': 0})

    # Do error prediction exercise was successfully predicted.
    error_prediction, error_confidence, error_message = classifier.predict_error(exercise, rows)

    if error_prediction == "Unknown":
        return jsonify({'status': 'FAILED', 'message': error_message})

    return jsonify({'status': 'OK',
                    'exercise': str(exercise),
                    'exercise_score': exercise_confidence,
                    'mistakes': error_prediction,
                    'mistakes_score': error_confidence})


if __name__ == '__main__':
    if not path.exists(config_file) or not path.isfile(config_file):
        print("Config file missing... (config.cfg)")

    pre_proc = pre_processing.PreProcessing()
    config = configparser.ConfigParser()
    config.read(config_file)

    classifier = classifier.Classifier(config)
    if classifier.load_models():
        app.run(port=int(config['WEB']['Port']), host=str(config['WEB']['Host']))



