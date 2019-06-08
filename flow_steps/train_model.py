import click
import mlflow
import numpy as np
import pandas as pd
from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


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


def prepare_train_data(train_data: pd.DataFrame, scaled_data: pd.DataFrame, left_bound: int) -> tuple:
    x_train, y_train = [], []

    for i in range(left_bound, len(train_data)):
        x_train.append(scaled_data[i - left_bound:i, 0])
        y_train.append(scaled_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train


def prepare_test_data(inputs: pd.DataFrame, left_train_bound: int) -> np.ndarray:
    x_test = []
    for i in range(left_train_bound, inputs.shape[0]):
        x_test.append(inputs[i - left_train_bound:i, 0])
    x_test = np.array(x_test)

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_test


@click.command(help="Train LSTM Network using transformed data")
@click.option("--ratings-data", type=str)
@click.option("--lstm-units", type=int, default=50)
def train_model(ratings_data, lstm_units):
    lstm_data = pd.read_csv(ratings_data)
    scalar = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scalar.fit_transform(lstm_data)

    with mlflow.start_run():
        train_data, test_data = train_test_split(lstm_data, test_size=0.2, shuffle=False)
        left_train_bound = int(len(train_data) * 0.25)

        x_train, y_train = prepare_train_data(train_data, scaled_data, left_train_bound)

        mlflow.log_metric("training_nrows", train_data.count())
        mlflow.log_metric("test_nrows", test_data.count())

        height = x_train.shape[1]
        model = LSTMNet.build(height, lstm_units)
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

        inputs = lstm_data[len(lstm_data) - len(test_data) - left_train_bound:].values
        inputs = inputs.reshape(-1, 1)
        inputs = scalar.transform(inputs)
        x_test = prepare_test_data(inputs, left_train_bound)

        predictions = model.predict(x_test)
        predictions = scalar.inverse_transform(predictions)
        r_mean_square = root_square_mean(test_data, predictions)
        mlflow.log_metric("train_rms", r_mean_square)

        print(f"The model had a RMS on the test set of {r_mean_square}")
        mlflow.keras.log_model(model, "keras-model")


if __name__ == '__main__':
    train_model()
