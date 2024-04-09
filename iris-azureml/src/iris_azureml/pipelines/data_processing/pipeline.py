"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node
from .nodes import read_data


def create_etl_pipeline(**kwargs) -> Pipeline:
    pipeline_etl = Pipeline(
        [
            node(
                func=read_data,
                inputs=['data'],
                outputs=['labeled_data'],
                tags=["etl_labels"]
                    )
        ]
    )
    return pipeline_etl