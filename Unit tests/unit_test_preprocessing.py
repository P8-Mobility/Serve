import unittest
from pre_processing import PreProcessing
import pandas as pd
import math


class TestPreProcessing(unittest.TestCase):

    def test_merged_rows_pass(self):
        pre_proc = PreProcessing()

        data_rows = [{"acc_x": 1.2, "acc_y": 3.4, "acc_z": 5.6, "gyro_x": 1.2, "gyro_y": 3.4, "gyro_z": 5.6,
                      "address": '123', "unix_time": 456, },
                     {"acc_x": 6.5, "acc_y": 4.3, "acc_z": 2.1, "gyro_x": 6.5, "gyro_y": 4.3, "gyro_z": 2.1,
                      "address": '321', "unix_time": 654, }]
        test_sensors = ['123', '321']

        d_result = {"0.acc_x": [1.2], "0.acc_y": [3.4], "0.acc_z": [5.6],
                    "0.gyro_x": [1.2], "0.gyro_y": [3.4], "0.gyro_z": [5.6],
                    "1.acc_x": [6.5], "1.acc_y": [4.3], "1.acc_z": [2.1],
                    "1.gyro_x": [6.5], "1.gyro_y": [4.3], "1.gyro_z": [2.1]}
        df_result = pd.DataFrame(data=d_result)

        self.assertTrue(df_result.equals(pre_proc.merged_rows(rows=data_rows, sensors=test_sensors)))

    def test_min_max_pass(self):
        pre_proc = PreProcessing()
        d_test = {'col1': [1, 2], 'col2': [3, 4]}
        df_test = pd.DataFrame(data=d_test)

        d_result = {'col1': [0.0, 1], 'col2': [0.0, 1]}
        df_result = pd.DataFrame(data=d_result)

        self.assertTrue(df_result.equals(pre_proc.min_max_normalization(data=df_test)))

    def test_min_max_fail(self):
        pre_proc = PreProcessing()
        d_test = {'col1': [1, 2], 'col2': [3, 4]}
        df_test = pd.DataFrame(data=d_test)

        self.assertFalse(df_test.equals(pre_proc.min_max_normalization(data=df_test)))

    def test_rolling_all_data_pass(self):
        pre_proc = PreProcessing()
        d_test = {'col1': [1, 2, 3], 'col2': [4, 5, 6]}
        df_test = pd.DataFrame(data=d_test)

        col1mean = (1+2+3)/3
        col2mean = (4+5+6)/3
        d_result = {'col1': [col1mean], 'col2': [col2mean]}
        df_result = pd.DataFrame(data=d_result, index=[2])

        self.assertTrue(df_result.equals(pre_proc.rolling_all_data(data=df_test, rolling_size=3)))

    def test_acc_mag_pass(self):
        pre_proc = PreProcessing()
        d_test = {'0.acc_x': [1, 2], '0.acc_y': [3, 4], '0.acc_z': [5, 6]}
        df_test = pd.DataFrame(data=d_test)

        result1 = math.sqrt(1 ** 2 + 3 ** 2 + 5 ** 2)
        result2 = math.sqrt(2 ** 2 + 4 ** 2 + 6 ** 2)

        df_assert = pre_proc.calculate_acceleration_magnitude(data=df_test, skip_indexes=[1, 2, 3, 4, 5, 6, 7, 8, 9])
        acc_mag1 = df_assert.iloc[0]['0.acc_magnitude']
        acc_mag2 = df_assert.iloc[1]['0.acc_magnitude']

        self.assertEqual([result1, result2], [acc_mag1, acc_mag2])

    def test_pop_tag_data_pass(self):
        pre_proc = PreProcessing()
        d_test = {'0.acc_x': [1], '0.acc_y': [2], '0.acc_z': [3], '0.gyro_x': [4],
                  '0.gyro_y': [5], '0.gyro_z': [6], 'foo': [7]}
        df_test = pd.DataFrame(data=d_test)

        d_result = {'foo': [7]}
        df_result = pd.DataFrame(data=d_result)

        self.assertTrue(df_result.equals(pre_proc.pop_tag_data(data=df_test, tag_indexes=[0])))

    def test_pop_tag_data_with_mag_pass(self):
        pre_proc = PreProcessing()
        d_test = {'0.acc_x': [1], '0.acc_y': [2], '0.acc_z': [3], '0.gyro_x': [4],
                  '0.gyro_y': [5], '0.gyro_z': [6], '0.acc_magnitude': [7]}
        df_test = pd.DataFrame(data=d_test)

        self.assertTrue(pre_proc.pop_tag_data(data=df_test, tag_indexes=[0]).empty)

    def test_sliding_windows_pass(self):
        pre_proc = PreProcessing()
        d_test = {'col1': [1, 2, 3], 'col2': [4, 5, 6]}
        df_test = pd.DataFrame(data=d_test)

        d_result1 = {'col1': [1, 2], 'col2': [4, 5]}
        d_result2 = {'col1': [2, 3], 'col2': [5, 6]}
        df_result1 = pd.DataFrame(data=d_result1)
        df_result2 = pd.DataFrame(data=d_result2, index=[1, 2])

        self.assertTrue(df_result1.equals(pre_proc.sliding_windows(data=df_test, window_size=2)[0]))
        self.assertTrue(df_result2.equals(pre_proc.sliding_windows(data=df_test, window_size=2)[1]))


if __name__ == '__main__':
    unittest.main()
