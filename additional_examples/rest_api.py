"""
Mlflow REST API example. Mlflow Rest API os pretty useeful for applications
where you do not want to embed whole library or connect many microservices written in
different languages.
More information and endpoint descriptions can be found here:
https://www.mlflow.org/docs/latest/rest-api.html

Usage:
Run: mlflow server for starting a service, then
run rst_api.py script. On UI you should see properly logged data.
Default ExperimentID = 0
"""

import time
from random import random

import requests


class MLFlowTrackingRestApi:
    def __init__(self, hostname: str = 'localhost', port: int = 5000, experiment_id: int = 0):
        self.base_url = 'http://' + hostname + ':' + str(port) + '/api/2.0/preview/mlflow'
        self.experiment_id = experiment_id
        self.run_id = self.create_run()

    def create_run(self):
        """Create a new run for tracking."""
        url = self.base_url + '/runs/create'
        payload = {'experiment_id': self.experiment_id,
                   'start_time': int(time.time() * 1000)}
        r = requests.post(url, json=payload)

        if r.status_code == 200:
            return r.json()['run']['info']['run_uuid']

        print("Creation of mlflow run failed!")
        return None

    def list_experiments(self) -> list:
        """Get all experiments."""
        url = self.base_url + '/experiments/list'
        r = requests.get(url)

        if r.status_code == 200:
            return r.json()['experiments']
        return []

    def set_run_tag(self, param: dict) -> int:
        url = self.base_url + '/runs/set-tag'
        payload = {'run_id': self.run_id, 'key': param['key'], 'value': param['value']}
        r = requests.post(url, json=payload)
        return r.status_code

    def log_param(self, param: dict) -> int:
        """Log a parameter dict for the given run."""
        url = self.base_url + '/runs/log-parameter'
        payload = {'run_id': self.run_id, 'key': param['key'], 'value': param['value']}
        r = requests.post(url, json=payload)
        return r.status_code

    def log_metric(self, metric: dict) -> int:
        """Log a metric dict for the given run."""
        url = self.base_url + '/runs/log-metric'
        payload = {'run_id': self.run_id, 'key': metric['key'], 'value': metric['value']}
        r = requests.post(url, json=payload)
        return r.status_code


if __name__ == "__main__":
    mlflow_rest = MLFlowTrackingRestApi()
    print("Mlflow REST API example. Choose action for list below.")
    print("[1] Log random parameter value")
    print("[2] Log random metric")
    print("[3] List local experiments")
    print("[4] Set 'XD' tag on current run")

    while True:
        action_num = int(input("Insert action number: "))
        if action_num == 1:
            mlflow_rest.log_param({'key': 'test_param', 'value': str(round(random(), 3))})
        elif action_num == 2:
            mlflow_rest.log_metric({'key': 'test_metric', 'value': round(random(), 3)})
        elif action_num == 3:
            experiments = mlflow_rest.list_experiments()
            print(experiments)
        elif action_num == 4:
            mlflow_rest.set_run_tag({'key': 'test_tag', 'value': "random_tag"})
        else:
            print("No action for given input")
