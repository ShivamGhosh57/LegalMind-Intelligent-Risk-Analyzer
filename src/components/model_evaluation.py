import os
import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
from joblib import dump

# Add the project root directory to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.exception import CustomException
from src.logger import logging

def eval_metrics(actual, pred):
    """
    Calculates and returns a dictionary of classification metrics.
    """
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred, average='weighted')
    recall = recall_score(actual, pred, average='weighted')
    f1 = f1_score(actual, pred, average='weighted')
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

if __name__ == '__main__':
    # Example usage
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 1, 0, 0, 1]
    metrics = eval_metrics(y_true, y_pred)
    print(metrics)
