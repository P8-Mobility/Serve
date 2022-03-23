import librosa
import os.path
import scipy.io.wavfile
import numpy as np


def load(path):
    if os.path.exists(path):
        time_series, sampling_rate = librosa.load(path, sr=None)
        return Audio(path, time_series, sampling_rate)
    else:
        return None


class Audio:
    def __init__(self, path, time_series, sampling_rate):
        folder, filename = os.path.split(path)
        self.__folder = folder
        self.__path = path
        self.__filename = filename
        self.time_series = time_series
        self.__sampling_rate = sampling_rate
        self.__original_duration = self.get_duration()
        self.__original_time_series = time_series
        self.features = np.ndarray

    @property
    def get_filename(self):
        return self.__filename

    @property
    def get_path(self):
        return self.__path

    @property
    def get_sampling_rate(self):
        return self.__sampling_rate

    @property
    def get_id(self):
        return self.__filename.split('-')[2].split('.')[0]

    @property
    def is_wrong(self):
        return 'wrong' in self.__filename

    def get_duration(self):
        return librosa.get_duration(y=self.time_series, sr=self.__sampling_rate)

    def get_orignial_time_series(self):
        return self.__original_time_series

    def get_original_duration(self):
        return self.__original_duration

    def __hash__(self):
        return hash(self.time_series)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.time_series == other.time_series
        return NotImplemented

    def save(self, path):
        data = np.array(self.time_series)
        scipy.io.wavfile.write(path, self.__sampling_rate, data.astype(np.int16))

    def mel_spectrogram(self):
        data = np.array(self.time_series, dtype=np.float32)
        D = np.abs(librosa.stft(data)) ** 2
        S = librosa.feature.melspectrogram(S=D, sr=self.get_sampling_rate)
        return S

    def stft(self, window_size=100, hop_length=100):
        window = np.hanning(window_size)
        stft_out = librosa.core.spectrum.stft(self.time_series, n_fft=window_size, hop_length=hop_length, window=window)
        out = 2 * np.abs(stft_out) / np.sum(window)
        return out

    def mfccs(self):
        data = np.array(self.time_series, dtype=np.float32)
        return librosa.feature.mfcc(y=data, sr=self.get_sampling_rate)


