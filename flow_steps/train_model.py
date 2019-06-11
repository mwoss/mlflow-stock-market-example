import math
from os import mkdir, path

import click
import mlflow
import numpy as np
import pandas as pd

from keras.callbacks import Callback
from keras.layers import Dense, LSTM
from keras.models import Sequential
from mlflow import keras, pyfunc
from mlflow.utils.file_utils import TempDir
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from constants import (TRAIN_ROWS_METRIC, TEST_ROWS_METRIC, RMS_METRIC, MODEL_ARTIFACT_NAME, MODEL_ARTIFACT_PATH,
                       STOCK_MODEL_PATHS)


class MLflowLogger(Callback):
    """
    Logger based on code from mlflow/example/flower_classifier.
    Keras callback for logging metrics and final model with MLflow.
    Metrics are logged after every epoch. The logger keeps track of the best model based on the
    validation metric. At the end of the training, the best model is logged with MLflow.
    """

    def __init__(self, model, artifact_path):
        super().__init__()
        self._model = model
        self._best_train_loss = math.inf
        self._artifact_path = artifact_path
        self._best_weights = None

    def on_epoch_end(self, epoch: int, logs=None):
        """
        Log train Keras metrics with MLflow. Update the best model if the model improved on the validation data.
        """
        if not logs:
            return

        for name, value in logs.items():
            name = "train_" + name
            mlflow.log_metric(name, value)
        train_loss = logs["loss"]
        if train_loss < self._best_train_loss:
            self._best_train_loss = train_loss
            self._best_weights = [x.copy() for x in self._model.get_weights()]

    def on_train_end(self, *args, **kwargs):
        """
        Log the best model with MLflow.
        """
        self._model.set_weights(self._best_weights)
        self.log_model(keras_model=self._model, artifact_path=self._artifact_path)

    @staticmethod
    def log_model(keras_model, artifact_path):
        """
        Log model to mlflow.
        :param keras_model: Keras model to be saved.
        :param artifact_path: Run-relative artifact path this model is to be saved to.
        """
        with TempDir() as tmp:
            data_path = tmp.path(STOCK_MODEL_PATHS)
            if not path.exists(data_path):
                mkdir(data_path)

            keras_path = path.join(data_path, MODEL_ARTIFACT_NAME)
            keras.save_model(keras_model, path=keras_path)
            pyfunc.log_model(artifact_path=artifact_path,
                             loader_module=__name__,
                             code_path=[__file__],
                             data_path=data_path)


class LSTMNet:
    @staticmethod
    def build(height: int, lstm_units: int):
        input_shape = (height, 1)

        model = Sequential()
        model.add(LSTM(lstm_units, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(lstm_units))
        model.add(Dense(1))

        return model


def root_square_mean(test_data: np.ndarray, predictions: np.ndarray) -> float:
    return np.sqrt(np.mean(np.power(test_data - predictions, 2)))


def prepare_data(raw_data: pd.DataFrame, left_bound: int, right_bound: int) -> tuple:
    x, y = [], []

    for i in range(left_bound, right_bound):
        x.append(raw_data[i - left_bound:i, 0])
        y.append(raw_data[i, 0])
    x, y = np.array(x), np.array(y)

    x = np.reshape(x, (x.shape[0], x.shape[1], 1))

    return x, y


@click.command(help="Train LSTM Network using transformed data")
@click.option("--stock-data", type=str)
@click.option("--lstm-units", type=int, default=50)
def train_model(stock_data, lstm_units):
    lstm_data = pd.read_csv(stock_data)
    scalar = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scalar.fit_transform(lstm_data)

    with mlflow.start_run(run_name="train"):
        train_data, test_data = train_test_split(lstm_data, test_size=0.2, shuffle=False)
        left_train_bound = int(len(train_data) * 0.25)

        mlflow.log_metric(TRAIN_ROWS_METRIC, float(train_data.shape[0]))
        mlflow.log_metric(TEST_ROWS_METRIC, float(test_data.shape[0]))

        x_train, y_train = prepare_data(scaled_data, left_train_bound, train_data.shape[0])

        inputs = lstm_data[len(lstm_data) - len(test_data) - left_train_bound:].values
        inputs = inputs.reshape(-1, 1)
        inputs = scalar.transform(inputs)
        x_test, y_test = prepare_data(inputs, left_train_bound, inputs.shape[0])

        height = x_train.shape[1]
        model = LSTMNet.build(height, lstm_units)
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        model.fit(x_train,
                  y_train,
                  epochs=1,
                  batch_size=1,
                  verbose=1,
                  callbacks=[MLflowLogger(model, artifact_path=MODEL_ARTIFACT_PATH)]
                  )

        predictions = model.predict(x_test)
        predictions = scalar.inverse_transform(predictions)
        r_mean_square = root_square_mean(test_data, predictions)
        mlflow.log_metric(RMS_METRIC, float(r_mean_square))

        print(f"The model had a RMS on the test set of {float(r_mean_square)}")
        print("Model and metrics uploaded. Done")


if __name__ == '__main__':
    train_model()
