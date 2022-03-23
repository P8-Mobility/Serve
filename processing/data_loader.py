import os

import numpy as np
import pandas as pd
import warnings
import librosa

from processing import audio, transformer
from processing.audio import Audio


class DataLoader:
    def __init__(self, audio_list: list[Audio] = None):
        if audio_list is None:
            audio_list = []

        self.__data = audio_list
        self.__duration_scale = 0
        self.__duration_sum = 0
        self.__settings = {"trim_threshold": 20, "mfcc": False, "scale_length": False}

    def clear(self):
        self.__data.clear()
        self.__duration_scale = 0
        self.__duration_sum = 0

    def add_folder_to_model(self, path: str):
        if os.path.isdir(path):
            for filename in os.listdir(path):
                if filename.endswith(".wav"):
                    self.add_file_to_model(path + "/" + filename)

    def add_file_to_model(self, path: str):
        if os.path.isfile(path):
            self.__data.append(audio.load(path))

    def load_file(self, path: str):
        if os.path.isfile(path):
            audio_file = audio.load(path)
            self.preprocessing(audio_file)
            if self.__settings.get("scale_length"):
                self.scale(audio_file)
            return audio_file
        return None

    def fit(self):
        for audio_file in self.__data:
            self.preprocessing(audio_file)
            self.__duration_sum += audio_file.get_duration()

        self.__duration_scale = self.__duration_sum / len(self.__data)

        if self.__settings.get("scale_length"):
            for audio_file in self.__data:
                self.scale(audio_file)

    def size(self):
        return len(self.__data)

    def get_data_files(self):
        return self.__data

    def get_as_dataframe(self) -> pd.DataFrame:
        file_names = []
        time_series_data = []

        for audio_file in self.__data:
            file_names.append(audio_file.get_filename)
            time_series_data.append(audio_file.time_series)

        return pd.DataFrame({"filename": file_names, "time_series": time_series_data})

    def preprocessing(self, audio_file: Audio):
        transformer.remove_noise(audio_file)
        transformer.normalize(audio_file)
        transformer.trim(audio_file, self.__settings.get("trim_threshold"))
        if self.__settings.get("mfcc"):
            transformer.mfccs(audio_file)

    def scale(self, audio_file: Audio):
        audio_file.time_series = librosa.effects.time_stretch(audio_file.time_series, rate=audio_file.get_duration() / self.__duration_scale)

    def store_processed_files(self, path: str):
        if os.path.isdir(path):
            for audio_file in self.__data:
                audio_file.time_series = np.array([int(s * 32768) for s in audio_file.time_series])
                audio_file.save(path + audio_file.get_filename)
        else:
            warnings.warn(f"Path must be a directory, {path} is not.")

    def change_setting(self, key: str, value: any):
        if self.__settings.get(key) is None:
            warnings.warn(f"They key, {key}, was not found in the settings dictionary, so default settings are used.")
        else:
            self.__settings[key] = value

