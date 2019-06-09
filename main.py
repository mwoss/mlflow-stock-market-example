from os import path

import click
import mlflow
from mlflow.entities import RunStatus
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils import mlflow_tags
from mlflow.utils.logging_utils import eprint


def _already_ran(entry_point_name, parameters, git_commit, experiment_id=None):
    """Best-effort detection of if a run with the given entrypoint name,
    parameters, and experiment id already ran. The run must have completed
    successfully and have at least the parameters provided.
    """
    experiment_id = experiment_id if experiment_id is not None else _get_experiment_id()
    client = mlflow.tracking.MlflowClient()
    all_run_infos = reversed(client.list_run_infos(experiment_id))
    for run_info in all_run_infos:
        full_run = client.get_run(run_info.run_id)
        tags = full_run.data.tags
        if tags.get(mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT, None) != entry_point_name:
            continue
        match_failed = False
        for param_key, param_value in parameters.items():
            run_value = full_run.data.params.get(param_key)
            if run_value != param_value:
                match_failed = True
                break
        if match_failed:
            continue

        if run_info.status != RunStatus.FINISHED:
            eprint(f"Run matched, but is not FINISHED. Skipping. Run_id={run_info.run_id}, status={run_info.status}")
            continue

        previous_version = tags.get(mlflow_tags.MLFLOW_GIT_COMMIT, None)
        if git_commit != previous_version:
            eprint(f"Run matched, but has a different source version. Skipping. "
                   f"Found={previous_version}, expected={git_commit}")
            continue
        return client.get_run(run_info.run_id)
    eprint("No matching run has been found.")
    return None


def _get_or_run(entrypoint, parameters, git_commit, use_cache=True):
    existing_run = _already_ran(entrypoint, parameters, git_commit)
    if use_cache and existing_run:
        print(f"Found existing run for entrypoint={entrypoint} and parameters={parameters}")
        return existing_run

    print(f"Launching new run for entrypoint={entrypoint} and parameters={parameters}")
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters)
    return mlflow.tracking.MlflowClient().get_run(submitted_run.run_id)


@click.command("Run entire flow, from downloading data to training model")
@click.option("--company-abbreviation", type=str, default="MSFT")
@click.option("--lstm-units", type=int, default=50)
@click.option("--max-row-limit", type=int, default=100000)
def workflow(company_abbreviation, lstm_units, max_row_limit):
    with mlflow.start_run() as active_run:
        git_commit = active_run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)
        download_data_run = _get_or_run("download_raw_data",
                                        {"company_abbreviation": company_abbreviation},
                                        git_commit)

        dataset_stock_csv = path.join(download_data_run.info.artifact_uri, "dataset-stock-dir", "dataset-market.csv")
        transform_data_run = _get_or_run("transform_data",
                                         {"dataset_stock_csv": dataset_stock_csv, "max_row_limit": max_row_limit},
                                         git_commit)
        transformed_dataset_dir = path.join(transform_data_run.info.artifact_uri, "transformed-dataset-dir")

        _get_or_run("train_model",
                    {"stock_data": transformed_dataset_dir, "hidden_units": lstm_units},
                    git_commit,
                    use_cache=False)


if __name__ == '__main__':
    workflow()
