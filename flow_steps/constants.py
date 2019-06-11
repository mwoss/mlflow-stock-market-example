QUANLD_API = "https://www.quandl.com/api/v3/datasets/WIKI"

# METRICS
TRAIN_ROWS_METRIC = "training_nrows"
TEST_ROWS_METRIC = "test_nrows"
RMS_METRIC = "train_rms"

# ARTIFACTS
DATASET_ARTIFACT_DIR = "dataset-stock-dir"
TRANSFORMED_ARTIFACT_DIR = "transformed-dataset-dir"
MODEL_ARTIFACT_PATH = "models"
MODEL_ARTIFACT_NAME = "keras-model"

# FILENAMES
DATASET_NAME = "dataset-market.csv"
TRANSFORMED_DATASET_NAME = "transformed-dataset.csv"

# FLOW STEPS
DOWNLOAD_STEP = "download_raw_data"
TRANSFORM_STEP = "transform_data"
TRAIN_STEP = "train_model"
