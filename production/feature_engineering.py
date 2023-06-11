"""Processors for the feature engineering step of the worklow.

The step loads cleaned training data, processes the data for outliers,
missing values and any other cleaning steps based on business rules/intuition.

The trained pipeline and any artifacts are then saved to be used in
training/scoring pipelines.
"""
import logging
import os.path as op
import mlflow
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from feature_engine.outliers import Winsorizer
from scripts import log_transformer, scaling_transform

from ta_lib.core.api import (
    load_dataset,
    register_processor,
    save_pipeline,
    DEFAULT_ARTIFACTS_PATH,
)

logger = logging.getLogger(__name__)


@register_processor("feat_engg", "transform_features")
def transform_features(context, params):
    """Transform dataset to create training datasets."""

    input_features_ds = "train/sales/features"
    input_target_ds = "train/sales/target"

    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    # load datasets
    train_X = load_dataset(context, input_features_ds)
    train_y = load_dataset(context, input_target_ds)

    # cat_columns = train_X.select_dtypes("object").columns
    # num_columns = train_X.select_dtypes("number").columns

    # Treating Outliers
    winsorizer = Winsorizer(
        capping_method=params["outliers"]["method"],
        tail=params["outliers"]["tail"],
        fold=params["outliers"]["fold"],
        variables=list(train_X.columns),
    )
    train_X = winsorizer.fit_transform(train_X)

    # # Dropping not required columns
    # train_X = train_X.drop(["year", "week_number"], axis=1)

    curated_columns = list(
        set(train_X.columns.to_list())
        - set(
            [
                "year",
                "week_number",
            ]
        )
    )

    # saving the list of relevant columns and the pipeline.
    save_pipeline(
        curated_columns, op.abspath(op.join(artifacts_folder, "curated_columns.joblib"))
    )

    train_X = train_X[curated_columns]
    train_y = np.log(1 + train_y).values.ravel()

    # Building Pipeline for the model
    features_transformer = Pipeline(
        [
            ("log_transformer", FunctionTransformer(log_transformer)),
            ("scaling", FunctionTransformer(scaling_transform)),
            # ("linear_regression", SKLStatsmodelOLS()),
        ]
    )

    # # Fitting the piepline
    # reg_ppln_ols = features_transformer.fit(train_X, train_y)

    # _ = features_transformer.fit_transform(train_X)
    _ = features_transformer.fit(train_X)

    # saving the pipeline.
    save_pipeline(
        features_transformer, op.abspath(op.join(artifacts_folder, "features.joblib"))
    )

    # Configuring mlflow
    mlflow.set_experiment("Sales Prediction")
    with mlflow.start_run(run_name="Feature Engineering"):
        mlflow.log_param("outliers_method", params["outliers"]["method"])
        mlflow.log_param("outliers_tail", params["outliers"]["tail"])
        mlflow.log_param("outliers_fold", params["outliers"]["fold"])

        # metric
        mlflow.log_metric("num_features", train_X.shape[1])

        # Log the artifacts
        # mlflow.log_artifact(input_features_ds, "input_features_ds")
        # mlflow.log_artifact(input_target_ds, "input_target_ds")
