import os
import sys
import numpy as np
import pandas as pd
def save_object(file_path: str, obj: object) -> None:
    """
    Save a Python object to a file using pickle.

    Args:
        file_path (str): The path where the object should be saved.
        obj (object): The Python object to be saved.

    Returns:
        None
    """
    import pickle
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise Exception(f"Error saving object to {file_path}: {e}") from e


def evaluate_model(X_train, y_train, X_test, y_test, models: dict) -> dict:
    """
    Train and evaluate multiple regression models and return a report of R2 scores.

    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        models (dict): Mapping name -> estimator instance

    Returns:
        dict: Mapping model name -> R2 score on test set
    """
    from sklearn.metrics import r2_score

    report = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = r2_score(y_test, preds)
            report[name] = score
        except Exception as e:
            # If a model fails to train/evaluate, record a very low score and continue
            report[name] = float('-inf')
    return report