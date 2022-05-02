from configparser import ConfigParser
from processing.audio import Audio
from pathlib import Path
import allosaurus.app as allo
import allosaurus.audio
import os


class Classifier:
    

    def __init__(self, config: ConfigParser, language):
        self.default_word_model = None
        self.other_models = {}
        self.config = config
        self.lang = language
        self.response_txt_dan = ""
        self.response_txt_ara = ""

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

    def prepare_feedback(self, actual_word, prediction):
        actual_phonemes = self.lang.word_phonemes(actual_word)
        response_dan = ""
        response_ara = ""

        if actual_phonemes == prediction:
            self.response_txt_dan = self.lang.get("dan", "correct")
            self.response_txt_ara = self.lang.get("ara", "correct")
        else:
            special_case_present = self.special_feedback(prediction, actual_word)
            if not special_case_present:
                self.response_txt_dan = self.lang.get("dan", "incorrect")
                self.response_txt_ara = self.lang.get("ara", "incorrect")

        return response_dan, response_ara

    def special_feedback(self, result, actual_word):
        special_feedback_cases = self.lang.get_special("dan")
        word_phonemes = (self.lang.word_phonemes(actual_word)).split
        result_phonemes = (self.lang.word_phonemes(result)).split
        for special_case in special_feedback_cases:
            special_index =
            if special_case.actual == :
                    pass
                    return True
        return False



