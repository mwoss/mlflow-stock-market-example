import csv
from os import path
from tempfile import mkdtemp as create_tmp_dir

import click
import mlflow
import requests

QUANLD_API = "https://www.quandl.com/api/v3/datasets/WIKI"


@click.command(help="Downloads the stock market dataset for given company. Saves it as an mlflow artifact")
@click.option("--company-abbreviation", type=str)
def download_csv(company_abbreviation: str):
    dataset_url = f"{QUANLD_API}/{company_abbreviation}"

    with mlflow.start_run():
        local_dir = create_tmp_dir()
        local_filename = path.join(local_dir, "dataset-market.csv")
        print(f"Downloading {dataset_url} to {local_filename}")

        dataset = requests.get(dataset_url)
        decoded_content = dataset.content.decode("utf-8").splitlines()

        with open(local_filename, "w", newline="") as file:
            writer = csv.writer(file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
            for line in decoded_content:
                columns = line.split(",")
                writer.writerow(columns)

        print(f"Uploading stock market data: {local_filename}")
        mlflow.log_artifact(local_filename, "dataset-stock-dir")


if __name__ == '__main__':
    download_csv()
