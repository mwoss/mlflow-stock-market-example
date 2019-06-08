from os import path
from tempfile import mkdtemp as create_tmp_dir

import click
import mlflow
import pandas as pd


def process_dataframe(df):
    data = df.sort_index(ascending=True, axis=0)
    processed_data = pd.DataFrame(index=range(0, len(df)), columns=["Date", "Close"])

    for i in range(0, len(data)):
        processed_data["Date"][i] = data["Date"][i]
        processed_data["Close"][i] = data["Close"][i]

    return processed_data

@click.command(help="Transform given CSV file. Strip unused columns and convert rest of data. "
                    "Transforms it into Parquet in an mlflow artifact called 'market-transformed-dir'")
@click.option("--dataset-stock-dir", type=str)
@click.option("--max-row-limit", type=int, default=10000, help="Limit the data size to run comfortably on slower pcs.")
def transform_data(dataset_stock_dir, max_row_limit):
    with mlflow.start_run():
        tmpdir = create_tmp_dir()
        transformed_dataset_dir = path.join(tmpdir, 'stock-dataset')
        print(f"Converting stock market data CSV {dataset_stock_dir}. Output: {transformed_dataset_dir}")

        df = pd.read_csv(dataset_stock_dir)
        lstm_data = process_dataframe(df)
        lstm_data.index = lstm_data.Date
        lstm_data.drop('Date', axis=1, inplace=True)

        if max_row_limit != -1:
            lstm_data = lstm_data[:max_row_limit]

        lstm_data.to_csv(path.join(transformed_dataset_dir, "transformed-dataset-dir.csv"), index=None, header=True)

        print(f"Uploading transformed dataset: {transformed_dataset_dir}")
        mlflow.log_artifacts(transformed_dataset_dir, "transformed-dataset-dir")


if __name__ == '__main__':
    transform_data()
