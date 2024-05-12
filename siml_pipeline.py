import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler

class Pipeline:
    def __init__(self, num_days=10):
        self.num_days = num_days
        self.scaler_ = None
        self.y_scaler_ = None

    def process(self, data: pd.DataFrame, y_label, split=0.75):
        split_idx = int(split * len(data))

        self.scaler_ = StandardScaler()
        self.y_scaler_ = StandardScaler()

        X = data.drop(columns=[y_label])
        Y = data[y_label]

        sliding_window = np.lib.stride_tricks.sliding_window_view
        # sliding window adds new axis at index 2 but it should be at 1
        X_train = sliding_window(
            self.scaler_.fit_transform(X.iloc[:split_idx]), self.num_days, axis=0).swapaxes(1,2)
        X_test = sliding_window(
            self.scaler_.transform(X.iloc[split_idx:]), self.num_days, axis=0).swapaxes(1,2)
        # remove last element from X as we want to predict next Y value
        X_train = X_train[:-1]
        X_test = X_test[:-1]

        # skip first elements as these cannot be predicted due to lack of input data
        Y_train = self.y_scaler_.fit_transform(Y.iloc[:split_idx].to_numpy().reshape(-1, 1))[self.num_days:]
        Y_test = self.y_scaler_.transform(Y.iloc[split_idx:].to_numpy().reshape(-1, 1))[self.num_days:]

        # shuffle
        shuffled_indexes = np.arange(Y_train.shape[0])
        np.random.shuffle(shuffled_indexes)

        return X_train[shuffled_indexes], X_test, Y_train[shuffled_indexes], Y_test

    def process_test(self, test_data: pd.DataFrame):
        sliding_window = np.lib.stride_tricks.sliding_window_view
        X = sliding_window(
            self.scaler_.fit_transform(test_data), self.num_days, axis=0).swapaxes(1,2)
        return X

    def restore(self, output):
        return self.y_scaler_.inverse_transform(output)
