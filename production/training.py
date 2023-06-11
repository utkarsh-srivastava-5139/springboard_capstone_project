"""Processors for the model training step of the worklow."""
import logging
import os.path as op
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from scripts import mape

from ta_lib.core.api import (
    load_dataset,
    load_pipeline,
    register_processor,
    save_pipeline,
    DEFAULT_ARTIFACTS_PATH,
)
from ta_lib.regression.api import SKLStatsmodelOLS

logger = logging.getLogger(__name__)


@register_processor("model_gen", "train_model")
def train_model(context, params):
    """Train a regression model."""
    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    input_features_ds = "train/sales/features"
    input_target_ds = "train/sales/target"

    # load training datasets
    train_X = load_dataset(context, input_features_ds)
    train_y = load_dataset(context, input_target_ds)

    # load pre-trained feature pipelines and other artifacts
    curated_columns = load_pipeline(op.join(artifacts_folder, "curated_columns.joblib"))
    features_transformer = load_pipeline(op.join(artifacts_folder, "features.joblib"))

    train_X = train_X[curated_columns]
    train_y = np.log(1 + train_y).values.ravel()

    # transform the training data
    train_X = pd.DataFrame(
        features_transformer.fit_transform(train_X),
    )

    # create training pipeline
    reg_ppln_ols = Pipeline([("estimator", SKLStatsmodelOLS())])

    # fit the training pipeline
    reg_ppln_ols.fit(train_X, train_y)

    # Obtain Training Metrics
    y_train_hat = reg_ppln_ols.predict(train_X)
    mse = mean_squared_error(train_y, y_train_hat)
    rmse = np.sqrt(mean_squared_error(train_y, y_train_hat))
    mae = mean_absolute_error(train_y, y_train_hat)
    mape_result = mape(train_y, y_train_hat)

    # Obtain the summary
    summary = reg_ppln_ols["estimator"].summary()

    # Save the summary to a temporary file
    summary_file = op.abspath(op.join(artifacts_folder, "ols_summary.txt"))
    with open(summary_file, "w") as f:
        f.write(summary.as_text())

    # save fitted training pipeline
    save_pipeline(
        reg_ppln_ols, op.abspath(op.join(artifacts_folder, "train_pipeline.joblib"))
    )

    # Configuring mlflow
    mlflow.set_experiment("Sales Prediction")
    with mlflow.start_run(run_name="Model Training"):
        mlflow.log_params(params)
        mlflow.log_param("Input Features", train_X.columns.to_list())
        mlflow.log_param("Input Target", "sales_units_value")

        # Log the model
        mlflow.sklearn.log_model(reg_ppln_ols, "reg_ppln_ols")

        mlflow.log_metric("Model - MAE", mae)
        mlflow.log_metric("Model - MSE", mse)
        mlflow.log_metric("Model - RMSE", rmse)
        mlflow.log_metric("Model - MAPE", mape_result)

        # Log the artifacts
        mlflow.log_artifact(op.join(artifacts_folder, "features.joblib"))
        mlflow.log_artifact(
            op.abspath(op.join(artifacts_folder, "train_pipeline.joblib"))
        )
        mlflow.log_artifact(op.abspath(op.join(artifacts_folder, "ols_summary.txt")))
