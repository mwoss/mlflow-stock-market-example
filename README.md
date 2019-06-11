#### MLFlow stock market prediction PoC 

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
You can compare the results or check your run metrics/artifacts using `mlflow ui`  
Change execution parameters using `-P` attribute, for example:

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

Each model log metrics/artifact that can be used by next flow step. All information about steps is saved in mlrun directory (artifacts, metadata etc).

Mlproject - the most important file in Mlflow app. The file that defines whole pipeline.  
Basic structure:
```text
name: [PIPELINE_NAME]
conda_env: [PATH TO yaml file with anaconda env]

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
 

Notes
----

* Files placed in flow_steps directory use relative imports due to Mlflow execution logic.
* PoC is based on mlflow/examples/multistep_workflow exmaple from mlflow repository.

Interesting and useful resources
----
*   [bleble](asd)