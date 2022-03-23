import librosa
from processing.audio import Audio
import numpy as np
import librosa.display
from numpy import dot
from numpy.linalg import norm
import noisereduce as nr


def trim(audio: Audio, db_threshold=30):
    audio.time_series, index = librosa.effects.trim(audio.time_series, top_db=db_threshold)


def stft(audio: Audio, window_size=100, hop_length=100):
    window = np.hanning(window_size)
    stft_out = librosa.core.spectrum.stft(audio.time_series, n_fft=window_size, hop_length=hop_length, window=window)
    out = 2 * np.abs(stft_out) / np.sum(window)
    #librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), y_axis='log', x_axis='time', sr=audio.get_sampling_rate)
    return out  # librosa.amplitude_to_db(out, ref=np.max)


def mfccs(audio: Audio):
    data = np.array(audio.time_series, dtype=np.float32)
    mfccs_features = librosa.feature.mfcc(y=data, sr=audio.get_sampling_rate)
    audio.features = np.mean(mfccs_features.T, axis=0)


def chroma_stft(audio: Audio):
    return librosa.feature.chroma_stft(y=audio.time_series, sr=audio.get_sampling_rate)


def amp_to_db(matrix):
    return librosa.amplitude_to_db(matrix, ref=np.max)


def normalize(audio: Audio):
    max_peak = np.max(np.abs(audio.time_series))
    ratio = 1 / max_peak
    audio.time_series = audio.time_series * ratio


def cos_similarity(vector1, vector2):
    v1_len = len(vector1)
    v2_len = len(vector2)

    if v1_len != v2_len:
        return None

    if v1_len > 0 and isinstance(vector1[0], list):
        sum_of_similarity = 0

        for i in range(v1_len):
            sum_of_similarity += dot(vector1[i], vector2[i]) / (norm(vector1[i]) * norm(vector2[i]))

        return sum_of_similarity / v1_len
    else:
        return dot(vector1, vector2) / (norm(vector1) * norm(vector2))


def remove_noise(audio: Audio):
    audio.time_series = nr.reduce_noise(y=audio.time_series, sr=audio.get_sampling_rate)


def stretch_to_same_time(audio1: Audio, audio2: Audio):
    audio1_duration = audio1.get_duration()
    audio2_duration = audio2.get_duration()
    avg = (audio1_duration+audio2_duration) / 2
    audio1.time_series = librosa.effects.time_stretch(audio1.time_series, rate=audio1_duration / avg)
    audio2.time_series = librosa.effects.time_stretch(audio2.time_series, rate=audio2_duration / avg)


def melspectrogram(audio: Audio):
    data = np.array(audio.time_series, dtype=np.float32)
    D = np.abs(librosa.stft(data)) ** 2
    S = librosa.feature.melspectrogram(S=D, sr=audio.get_sampling_rate)
    #mel_sgram = librosa.amplitude_to_db(S, ref=np.min)
    return S


