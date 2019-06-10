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


Notes
----

* Files placed in flow_steps directory use relative imports due to Mlflow execution logic.
* PoC is based on mlflow/examples/multistep_workflow exmaple from mlflow repository.