import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class Pipeline:
    def __init__(self, num_days=10):
        self.num_days = num_days
        self.scaler_ = None
        self.y_scaler_ = None

    @staticmethod
    def _sliding_window(array, window_size: int):
        # sliding window adds new axis at index 2 but it should be at 1
        return np.lib.stride_tricks.sliding_window_view(array, window_size, axis=0).swapaxes(1, 2)

    @staticmethod
    def _shuffled_index(array):
        idx = np.arange(array.shape[0])
        np.random.shuffle(idx)
        return idx

    def process(self, data: pd.DataFrame, y_label, split=0.75):
        split_idx = int(split * len(data))

        self.scaler_ = StandardScaler()
        self.y_scaler_ = StandardScaler()

        X = data.drop(columns=[y_label])
        Y = data[y_label]

        X_train = self._sliding_window(self.scaler_.fit_transform(X.iloc[:split_idx]), self.num_days)
        X_test = self._sliding_window(self.scaler_.transform(X.iloc[split_idx:]), self.num_days)
        # remove last element from X as we want to predict next Y value
        X_train = X_train[:-1]
        X_test = X_test[:-1]

        # skip first elements as these cannot be predicted due to lack of input data
        Y_train = self.y_scaler_.fit_transform(Y.iloc[:split_idx].to_numpy().reshape(-1, 1))[self.num_days:]
        Y_test = self.y_scaler_.transform(Y.iloc[split_idx:].to_numpy().reshape(-1, 1))[self.num_days:]

        shuffled_indexes = self._shuffled_index(Y_train)
        return X_train[shuffled_indexes], X_test, Y_train[shuffled_indexes], Y_test

    def process_test(self, test_data: pd.DataFrame):
        X = self._sliding_window(self.scaler_.transform(test_data), self.num_days)
        return X

    def restore(self, output):
        return self.y_scaler_.inverse_transform(output)

class SingleFeaturePipeline(Pipeline):
    def __init__(self, num_days=10, look_ahead=8):
        super().__init__(num_days)
        self.look_ahead = look_ahead

    def process(self, y_series, split=0.75):
        split_idx = int(split * len(y_series))

        self.scaler_ = StandardScaler()
        self.y_scaler_ = self.scaler_

        assert self.num_days >= self.look_ahead, "Look ahead cannot be larger than number look back of days"
        X = self._sliding_window(
            self.scaler_.fit_transform(y_series[:-self.look_ahead].values.reshape(-1, 1)), self.num_days)
        # truncate Y sliding window data because a smaller window will always produce more samples than needed
        Y = self._sliding_window(
            self.scaler_.transform(y_series[self.look_ahead:].values.reshape(-1, 1)), self.look_ahead)[:X.shape[0]]

        X_train = X[:split_idx]
        X_test = X[split_idx:]
        Y_train = Y[:split_idx]
        Y_test = Y[split_idx:]

        shuffled_idx = self._shuffled_index(Y_train)
        return X_train[shuffled_idx], X_test, Y_train[shuffled_idx], Y_test
