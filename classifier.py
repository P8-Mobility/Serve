from configparser import ConfigParser
from processing.audio import Audio
from pathlib import Path
import allosaurus.app as allo
import allosaurus.audio
import os


class Classifier:

    def __init__(self, config: ConfigParser):
        self.default_word_model = None
        self.other_models = {}
        self.config = config

    def load_models(self) -> bool:
        self.default_word_model = allo.read_recognizer(alt_model_path=Path('models/' + self.config['MODELS']['word_model']))

        for dirname in os.listdir("models/"):
            if dirname != self.config['MODELS']['word_model'] and dirname != ".gitkeep":
                self.other_models[dirname] = allo.read_recognizer(alt_model_path=Path('models/' + dirname))

        return self.default_word_model is not None

    def predict_word(self, audio_file: Audio, model: str = ""):
        if model != "" and self.other_models.__contains__(model):
            prediction = self.other_models[model].recognize(allosaurus.audio.Audio(audio_file.time_series, audio_file.get_sampling_rate))
        else:
            model = self.config['MODELS']['word_model']
            prediction = self.default_word_model.recognize(allosaurus.audio.Audio(audio_file.time_series, audio_file.get_sampling_rate))

        return prediction, model
