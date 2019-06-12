"""
Example showing how we can manage runs/experiments using mlflow library.
https://mlflow.org/docs/latest/python_api/mlflow.tracking.html
"""
from mlflow.tracking import MlflowClient

if __name__ == '__main__':
    client = MlflowClient()  # Mlflow Tracking Server client - create and manage run and experiments

    experiments = client.list_experiments()  # Get list of Experiment objects
    experiment_id = experiments[0].experiment_id  # Every experiment (grouped runs) has own unique id

    single_run = client.create_run(experiment_id)  # Crate new run, we can do exact same thing with experiments

    client.log_param(single_run.info.run_id, "key", "value")  # Set metrics, tags, artifacts, parameters, etc.
    client.log_metric(single_run.info.run_id, "key_metric", 1.33)
    client.set_tag(single_run.info.run_id, "key_tag", "tag")

    client.set_terminated(single_run.info.run_id)  # Finish run
