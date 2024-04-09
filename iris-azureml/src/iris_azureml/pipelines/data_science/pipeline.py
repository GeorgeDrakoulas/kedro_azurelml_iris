"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.14
"""
from kedro.pipeline import Pipeline, node

from .nodes import relabel_y, evaluate_model, feature_engineering_x, feature_engineering_y, split_data, train_xgb_model


def create_ml_pipeline(**kwargs) -> Pipeline:
    pipeline_labels = Pipeline(
            [
                node(
                    func=feature_engineering_x,
                    inputs=["labeled_data", "features"],
                    outputs=["X_data"],
                    tags=["training"],
                ),
                node(
                    func=feature_engineering_y,
                    inputs=["labeled_data"],
                    outputs=["Y_data"],
                    tags=["training"],
                ),
                node(
                    func=relabel_y,
                    inputs=['Y_data'],
                    outputs=['Y_data_re'],
                    tags=["training"],
                ),
                node(
                    func = split_data,
                    inputs = ['X_data', 'Y_data_re'],
                    outputs = ['X_train', 'X_test', 'y_train', 'y_test'],
                    tags=["training"],
                ),
                node(
                    func=train_xgb_model,
                    inputs=["X_train", "y_train", "automl_config"],
                    outputs="classifier",
                    name="train_model_node",
                ),
                node(
                    func=evaluate_model,
                    inputs=["classifier", "X_test", "y_test"],
                    outputs=None,
                    name="evaluate_model_node",
                ),
            ]
    )

