from configparser import ConfigParser
from processing.audio import Audio
from pathlib import Path
import allosaurus.app as allo
import allosaurus.audio


class Classifier:

    def __init__(self, config: ConfigParser):
        self.word_model = None
        self.config = config

    def load_models(self) -> bool:
        self.word_model = allo.read_recognizer(alt_model_path=Path('models/' + self.config['MODELS']['word_model']))
        return self.word_model is not None

    def predict_word(self, audio_file: Audio) -> str:
        return self.word_model.recognize(allosaurus.audio.Audio(audio_file.time_series, audio_file.get_sampling_rate))
