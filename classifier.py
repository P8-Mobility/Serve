from configparser import ConfigParser
from processing.audio import Audio


class Classifier:

    def __init__(self, config: ConfigParser):
        self.word_model = None
        self.config = config

    def load_models(self):
        # TODO: Load all models
        return True

    def predict_word(self, audio_file: Audio):
        return True

