MLFlow stock market prediction PoC
----

Stock market prediction - machine learning pipeline with MlFlow.  
Repository contains POC of machine learning pipeline using MlFlow library, PoC was based on multistep mlflow example from origin repository.  

MLflow is an open source platform to manage the ML lifecycle, including experimentation, reproducibility and deployment.  
Mlflow consists of three components:
* Mlflow tracking - experiment tracking module
* Mlflow projects - reproducible runs
* Mlflow model - model packaging  

ML lifecycle = **(-> raw data -> data preparation -> model training -> deployment -> raw_data -> ...)**
MLflow aims to take any codebase written in its format and make it reproducible and reusable by multiple data scientists.  


Getting started 
----
In order to use mlflow you have to setup python environment with requirements stored in `pre_requirements.txt`.  
```
<setup your new environment using your favourite tooo>
pip install -r pre_requirements.txt
```
Running mlflow pipeline is pretty straight forward. Run `mlflow run` command from project directory and that's all.
Mlflow will execute pipline definied in `MLproject` file.

```bash
mlflow run .
```
```bash
mlflow run git@github.com:mwoss/mlflow-stock-market-example.git [Yes, you run flows via github uri :3]
```
By default mlflow gather all local pipeline execution into one experiment (group), which can be useful
for comparing runs intended to tackle a particular task. In order to create new group use mlflow CLI: 
```bash
export MLFLOW_EXPERIMENT_NAME=new-experiment

mlflow experiments create --experiment-name new-experiemnt [this arguemnt is optional if you export above env var]
```

You can also compare the results or check your run metrics/artifacts using `mlflow ui`  
Parametrized runs can be executed using `-P` attribute, for example:

```bash
mlflow run . -P lstm_units=60
```
You can also run whole application using main script, just simply execute main.py
```bash
python main.py
```
Example overview
----
Starting from data_analysis directory. It contains jupyter notebook with a few different stock market prediction (linear regression, LSTM, moving avarage, knn).
I've ended up using LSTM. The last chart shows how well LSTM copes with the prediction on given Microsoft dataset.  

After choosing a proper prediction approach I've started defining pipeline.
Pipeline got split into 3 individual steps:
* download_raw_data - download dataset using Quanld API
* transform_data - prepare data for training purpose
* train_model - train LSTM network and upload model
* deploy_model - deploy trained model on S3, this step require aws credentials (optional step, can be removed from pipeline)

Each model log metrics/artifact that can be used by next flow step. All information about steps is saved in mlruns directory (artifacts, metadata etc).  
If you want to log metrics/artifacts etc. to remote servers, you can do it easily by setting MLFLOW_TRACKING_URI env variable or
using `mlflow.set_tracking_uri()` (more info here: [Where run are recorded](https://mlflow.org/docs/latest/tracking.html?fbclid=IwAR0E3Ozpn52sNheoW7OmS3GkYf0iOBVgoxOB8cKI-iQKbo2hK-tBGEjUSpA#where-runs-are-recorded))

**Mlproject file schema**  
Mlproject file it's probably the most important file in Mlflow app. The file that defines whole pipeline.  
Basic structure:
```text
name: [PIPELINE_NAME]
conda_env: [PATH TO yaml file with anaconda env]
docker_env:
    image: [IMAGE_NAME] (conda_env or docker_env, not both at once) 

entry_points:
    step1:
        parameters:
            PARAMETER_NAME: {type: TYPE, default: DEFAULT}
        command: "COMMAND TO EXECUTE"
    step2:
        parameters:
            PARAMETER_NAME1: TYPE
            PARAMETER_NAME2: {type: TYPE, default: DEFAULT}
        command: "COMMAND TO EXECUTE"

```

`conda.yml` contains requirements for project virtual environment. Mlflow creates venv on its own with given requirements file and run whole multistep flow within it.
 
Beyond example
----
In `additional_examples` directory I cover cool functionalities beyond this simple multistep pipeline.
* `experiments_management.py` - short demonstration of MlflowClient's capabilities
* `rest_api.py` - simple REST API example, useful for services written in different languages

Notes
----
PoC is based on mlflow/examples/multistep_workflow exmaple from mlflow repository.

Files placed in flow_steps directory use relative imports due to Mlflow execution logic.

Interesting and useful resources
----
* [Mlflow documentation](https://mlflow.org/docs/latest/index.html)
* [Mlflow overview - Spark AI Summit 2019](https://www.youtube.com/watch?v=QJW_kkRWAUs)
* [Complete Machine Learning Lifecycle with MLflow - workshop](https://www.youtube.com/watch?v=VVnCyPOlrbk)
* [How to Utilize MLflow and Kubernetes](https://www.youtube.com/watch?v=cDtzu4WBzWA)
* [Bunch of Mlflow examples from Spark Summit 2019](https://github.com/amesar/mlflow-spark-summit-2019)
* [Mlflow quickstart - Python](https://docs.azuredatabricks.net/_static/notebooks/mlflow/mlflow-quick-start-python.html)
* [Mlflow quickstart - Scala](https://docs.azuredatabricks.net/_static/notebooks/mlflow/mlflow-quick-start-scala.html)