"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.14
"""

import logging
from typing import Dict, Tuple
from typing import TextIO
from xgboost import XGBClassifier
import xgboost  as xgb
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from flaml import AutoML
import mlflow
from sklearn.metrics import accuracy_score

T = TypeVar("T")

logger = logging.getLogger(__name__)


def feature_engineering_x(data: pd.DataFrame, features: pd.DataFrame) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    X = data[features['features']]
    return X

def feature_engineering_y(data: pd.DataFrame) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    Y = data["species"]
    return Y


def relabel_y(y_data: pd.Series) -> pd.Series:
    # Convert string labels to integers
    label_map = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
    y_train_re = pd.Series([label_map[label] for label in y_data])
    return y_train_re

def split_data(x_data, y_data):
    X_train, X_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=10
    )
    return X_train, X_test, y_train, y_test

def train_xgb_model( x_train,
    X_train: pd.DataFrame, y_train: pd.Series, automl_config: str):


    mlflow.sklearn.autolog()
    automl = AutoML()
    automl_config = automl_config["automl_settings"]
    automl.fit(X_train, y_train, **automl_config)

    return automl

def evaluate_model(automl: T, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Calculates and logs the coefficient of determination.

      Args:
        regressor: Trained model.
        X_test: Testing data of the independent features
        y_test: Testing data of dependent feautures
    """
    y_pred = automl.predict(X_test)
    test_accuracy  = accuracy_score(y_test, y_pred)
    print(test_accuracy )
    print("Test Accuracy:", test_accuracy)
    mlflow.log_metric('accuracy', test_accuracy)

