import json
from configparser import ConfigParser
from os import path
import joblib
import pandas as pd
from statistics import mode
from collections import Counter
import torch
import cnn
import pre_processing


class Classifier:

    def __init__(self, config: ConfigParser):
        self.exercise_model = None
        self.error_models = {}
        self.config = config
        self.pre_proc = pre_processing.PreProcessing()

    def load_models(self):
        if path.exists("models/" + str(self.config['EXERCISE_CLASSIFIER']['Model'])):
            self.exercise_model = joblib.load("models/" + str(self.config['EXERCISE_CLASSIFIER']['Model']))
            self.exercise_model.set_params(n_jobs=int(self.config['EXERCISE_CLASSIFIER']['NumberOfThreads']),
                                           verbose=bool(self.config["EXERCISE_CLASSIFIER"]["Verbose"]))

            #Load error detection models
            error_models = json.loads(self.config['ERROR_CLASSIFIER']['Models'])

            for model in error_models:
                print("Loading model: "+model)
                if path.exists("models/" + model):
                    exercise_uuid = model.split('.')[0]

                    net = cnn.ConvNet().float()
                    net.load_state_dict(torch.load("models/" + model, map_location=torch.device('cpu')))
                    net.eval()

                    self.error_models[exercise_uuid] = net
                else:
                    print("Model missing...")
                    return False
        else:
            print("Missing model for classifying exercises... ")
            return False

        return True

    def predict_exercise(self, df: pd.DataFrame):
        skipped_tag_indexes = [2, 3]

        if len(df.index) < 10:
            return "Unknown", 0, "Not enough data to perform prediction..."

        # Do required preprocessing
        df = self.pre_proc.pop_tag_data(df, skipped_tag_indexes)
        df = self.pre_proc.rolling_all_data(df)
        df = self.pre_proc.calculate_acceleration_magnitude(df, skipped_tag_indexes)

        try:
            prediction = self.exercise_model.predict(df)
        except ValueError as e:
            return "Unknown", 0, str(e)

        count_predictions = Counter(prediction)
        prediction_value = mode(prediction)
        confidence = count_predictions[prediction_value] / len(df.index) * 100

        return prediction_value, confidence, ""

    def predict_error(self, exercise: str, df: pd.DataFrame):

        # Do required preprocessing
        df = self.pre_proc.min_max_normalization(df)
        windows = self.pre_proc.sliding_windows(df)

        # Retrieve the model for the specific exercise
        if exercise in self.error_models.keys():
            model = self.error_models.get(exercise)

            tensor_data = []
            for index, dataframe in enumerate(windows):
                data_tensor = torch.tensor(dataframe.values)
                tensor_data.append(data_tensor)

            classifications = []
            for window in tensor_data:
                window = window.unsqueeze(0).unsqueeze(0)
                classifications.append(torch.argmax(model(window.float()), 1).tolist()[0])

            count_predictions = Counter(classifications)
            prediction_value = mode(classifications)
            confidence = count_predictions[prediction_value] / len(df.index) * 100

            return prediction_value, confidence, ""
        else:
            return "Unknown", 0, "Exercise uuid does not exist..."
