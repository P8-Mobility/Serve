import math
from queue import Queue
from typing import List, Dict
import pandas as pd
from typing import cast


class PreProcessing:

    def merged_rows(self, rows: List, sensors: List[str]):
        """
        Merges the raw DataRows into a DataFrame consisting of rows with data from each sensor specified in sensors

        :param rows: List of DataRows with raw data to be merged
        :param sensors: List of sensor addresses used when validating merging
        :return: DataFrame with rows of data for each sensor at a time_step
        """
        dataframe_columns = self.__get_dataframe_columns_for_merged_rows(sensors)
        dataframe_rows = []

        sensor_lists = self.__initialize_sensor_dictionary(rows, sensors)

        while self.__all_lists_non_empty(sensor_lists):
            row_with_data_from_all_sensors = self.__get_full_row(sensors, sensor_lists)
            if row_with_data_from_all_sensors is None:
                continue

            dataframe_rows.append(row_with_data_from_all_sensors)

        return pd.DataFrame(dataframe_rows, columns=dataframe_columns)

    def __initialize_sensor_dictionary(self, rows: List, sensor_addresses: List[str]):
        """
        Initializes a dictionary with lists of DataRows for all sensors and fills in data from the DataRows

        :param rows: List of raw DataRows
        :param sensor_addresses: List of sensor addresses

        :return: The filled dictionary with full lists
        """
        sensor_lists: Dict[str, list] = {}

        for address in sensor_addresses:
            sensor_lists[address] = []

        # Fill lists with raw data from related sensor
        for row in rows:
            sensor_lists[row['address']].append(row)

        for address in sensor_addresses:
            sensor_lists[address].sort(key=self.__list_sort, reverse=True)

        return sensor_lists

    def __get_dataframe_columns_for_merged_rows(self, sensor_addresses: List[str]):
        """
        Generates a list of columns for the DataFrame

        :param sensor_addresses: List of data rows with data to include
        :return: List of columns for the DataFrame and list of columns with sensor data
        """
        columns = []

        for i, address in enumerate(sensor_addresses):
            columns.extend([str(i)+".acc_x", str(i)+".acc_y", str(i)+".acc_z", str(i)+".gyro_x", str(i)+".gyro_y", str(i)+".gyro_z"])

        return columns

    def __get_full_row(self, sensors, sensor_lists):
        """
        Merges data from the dictionary lists into a row with data from each sensor,
        removing any encountered data that does not have related data from every other sensor

        :return: List with the data from every sensor for a time_step
        """

        queue = Queue()
        the_full_row = []

        # Place all sensor addressed in the queue to ensure that data is retrieved from all sensors
        for sensor_address in sensors:
            queue.put(sensor_address)

        # Get data while data is still missing for a sensor (i.e. the queue is non-empty)
        while not queue.empty():
            sensor_address = queue.get()

            if len(sensor_lists[sensor_address]) == 0:
                return None

            # Take the last element from associated sensor data list
            sensor_row = cast(dict, sensor_lists[sensor_address].pop())

            the_full_row.extend([sensor_row['acc_x'], sensor_row['acc_y'], sensor_row['acc_z'],
                                 sensor_row['gyro_x'], sensor_row['gyro_y'], sensor_row['gyro_z']])

        return the_full_row

    def __list_sort(self, e: dict):
        return e['unix_time']

    def __all_lists_non_empty(self, sensor_lists: Dict[str, list]):
        """
        :param sensor_lists: Dictionary with sensor addresses and list of DataRows
        :return: False if any list is empty, True if none are
        """
        for current_list in sensor_lists.values():
            if len(current_list) == 0:
                return False
        return True

    def min_max_normalization(self, data: pd.DataFrame):
        """
        Performs min-max normalization on the DataFrame

        :param data: DataFrame to normalize
        :return: The normalized DataFrame
        """
        normalized_df = data.copy()
        for feature_name in data.columns:
            max_value = data[feature_name].max()
            min_value = data[feature_name].min()
            normalized_df[feature_name] = (data[feature_name] - min_value) / (max_value - min_value)

        return normalized_df

    def rolling_all_data(self, data: pd.DataFrame, rolling_size=10):
        rolled = data.rolling(rolling_size).mean()
        return rolled.iloc[rolling_size - 1:]

    def calculate_acceleration_magnitude(self, data: pd.DataFrame, skip_indexes: []):
        columns = data.columns.tolist()
        the_rows = data.values.tolist()

        for x in range(0, 10):
            if x in skip_indexes:
                continue
            columns.append(str(x) + '.acc_magnitude')

        for row in the_rows:  # It works!
            for x in range(0, 10):
                if x in skip_indexes:
                    continue

                acc_x = row[columns.index(str(x) + '.acc_x')]
                acc_y = row[columns.index(str(x) + '.acc_y')]
                acc_z = row[columns.index(str(x) + '.acc_z')]
                row.append(math.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2))

        return pd.DataFrame(the_rows, columns=columns)

    def pop_tag_data(self, data: pd.DataFrame, tag_indexes: []):

        for index in tag_indexes:
            for coord in ['x', 'y', 'z']:
                data.pop(str(index) + '.acc_' + coord)
                data.pop(str(index) + '.gyro_' + coord)
            if data.columns.__contains__(str(index) + '.acc_magnitude'):
                data.pop(str(index) + '.acc_magnitude')

        return data

    def sliding_windows(self, data: pd.DataFrame, window_size: int = 10):
        windows: List[pd.DataFrame] = []

        half_window_size = window_size / 2
        number_of_windows = int(len(data.index) / window_size * 2)

        i = 0
        while i < number_of_windows:
            windows.append(data.iloc[int(i * half_window_size):int(i * half_window_size + window_size)])
            i += 1
        else:
            last_window = windows.pop()
            if len(last_window.values) == window_size:
                windows.append(last_window)

        return windows
