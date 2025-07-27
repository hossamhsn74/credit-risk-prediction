"""
ML pipeline for credit risk prediction.

This module provides functions for:
- Loading and cleaning data
- Feature engineering
- Preprocessing (encoding and scaling)
- Training and evaluating models
- Saving and loading model artifacts
"""

import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score, accuracy_score, recall_score, confusion_matrix
import joblib
import os


def load_data(filepath):
    """
    Load the dataset from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_csv(filepath)


def clean_data(data):
    """
    Clean the data by removing outliers and missing values.

    Args:
        data (pd.DataFrame): Raw data.

    Returns:
        pd.DataFrame: Cleaned data.
    """
    data = data.dropna(axis=0)
    data = data[data['person_age'] <= 80]
    data = data[data['person_emp_length'] <= 60]
    return data


def feature_engineering(data):
    """
    Add engineered features to the data.

    Args:
        data (pd.DataFrame): Cleaned data.

    Returns:
        pd.DataFrame: Data with new features.
    """
    data['age_group'] = pd.cut(data['person_age'],
                               bins=[20, 26, 36, 46, 56, 66],
                               labels=['20-25', '26-35', '36-45', '46-55', '56-65'])
    data['income_group'] = pd.cut(data['person_income'],
                                  bins=[0, 25000, 50000, 75000,
                                        100000, float('inf')],
                                  labels=['low', 'low-middle', 'middle', 'high-middle', 'high'])
    data['loan_amount_group'] = pd.cut(data['loan_amnt'],
                                       bins=[0, 5000, 10000,
                                             15000, float('inf')],
                                       labels=['small', 'medium', 'large', 'very large'])
    data['loan_to_income_ratio'] = data['loan_amnt'] / data['person_income']
    data['loan_to_emp_length_ratio'] = data['person_emp_length'] / data['loan_amnt']
    data['int_rate_to_loan_amt_ratio'] = data['loan_int_rate'] / data['loan_amnt']
    return data


def preprocess_data(data, ohe=None, scaler=None, fit=True):
    """
    Preprocess the data: one-hot encode categoricals and scale numerics.

    Args:
        data (pd.DataFrame): Data to preprocess.
        ohe (OneHotEncoder): Fitted encoder or None.
        scaler (StandardScaler): Fitted scaler or None.
        fit (bool): Whether to fit the encoders/scalers.

    Returns:
        pd.DataFrame: Preprocessed data.
        OneHotEncoder: Fitted encoder.
        StandardScaler: Fitted scaler.
    """
    ohe_columns = [
        'cb_person_default_on_file', 'loan_grade', 'person_home_ownership',
        'loan_intent', 'income_group', 'age_group', 'loan_amount_group'
    ]
    scale_cols = [
        'person_income', 'person_age', 'person_emp_length', 'loan_amnt', 'loan_int_rate',
        'cb_person_cred_hist_length', 'loan_percent_income', 'loan_to_income_ratio',
        'loan_to_emp_length_ratio', 'int_rate_to_loan_amt_ratio'
    ]

    # One-hot encoding
    if fit:
        ohe = OneHotEncoder(handle_unknown='ignore')
        ohe.fit(data[ohe_columns])
    ohe_feature_names = ohe.get_feature_names_out(ohe_columns)
    ohe_data = pd.DataFrame(
        ohe.transform(data[ohe_columns]).toarray(),
        columns=ohe_feature_names,
        index=data.index
    )
    X_new = pd.concat([data.drop(ohe_columns, axis=1), ohe_data], axis=1)

    # Scaling
    if fit:
        scaler = StandardScaler()
        scaler.fit(X_new[scale_cols])
    X_new[scale_cols] = scaler.transform(X_new[scale_cols])

    return X_new, ohe, scaler


def train_models(X_train, y_train):
    """
    Train multiple classifiers.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.

    Returns:
        dict: Trained models.
    """
    models = {
        'KN': KNeighborsClassifier(),
        'xgb': XGBClassifier(),
        'cat': CatBoostClassifier(verbose=0),
        'lgb': LGBMClassifier()
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a model and return metrics.

    Args:
        model: Trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.

    Returns:
        dict: Metrics.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity
    }


def save_artifacts(model, ohe, scaler, prefix="catboost"):
    """
    Save the trained model, encoder, and scaler to disk.

    Args:
        model: Trained model.
        ohe: OneHotEncoder.
        scaler: StandardScaler.
        prefix (str): Prefix for filenames.
    """
    os.makedirs("model_artifacts", exist_ok=True)
    joblib.dump(model, f"model_artifacts/{prefix}_model.pkl")
    joblib.dump(ohe, f"model_artifacts/{prefix}_ohe.pkl")
    joblib.dump(scaler, f"model_artifacts/{prefix}_scaler.pkl")


def load_artifacts(prefix="catboost"):
    """
    Load the trained model, encoder, and scaler from disk.

    Args:
        prefix (str): Prefix for filenames.

    Returns:
        tuple: (model, ohe, scaler)
    """
    model = joblib.load(f"model_artifacts/{prefix}_model.pkl")
    ohe = joblib.load(f"model_artifacts/{prefix}_ohe.pkl")
    scaler = joblib.load(f"model_artifacts/{prefix}_scaler.pkl")
    return model, ohe, scaler


def main():
    """
    Main function to run the ML pipeline:
    - Loads and cleans data
    - Performs feature engineering and preprocessing
    - Trains and evaluates models
    - Saves the CatBoost model and preprocessing artifacts
    """
    # Load and prepare data
    data = load_data("data/credit_risk_dataset.csv")
    data = clean_data(data)
    data = feature_engineering(data)

    # Split features and target
    X = data.drop(['loan_status'], axis=1)
    y = data['loan_status']
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=12
    )

    # Preprocess
    X_new, ohe, scaler = preprocess_data(x_train, fit=True)
    X_new_test, _, _ = preprocess_data(
        x_test, ohe=ohe, scaler=scaler, fit=False
    )

    # Train models
    models = train_models(X_new, y_train)
    # Evaluate
    for name, model in models.items():
        metrics = evaluate_model(model, X_new_test, y_test)
        print(f"Model: {name}")
        for k, v in metrics.items():
            print(f"  {k.capitalize()}: {v:.4f}")
        print()
    # Save CatBoost model, encoder, and scaler for FastAPI
    save_artifacts(models['cat'], ohe, scaler, prefix="catboost")


if __name__ == "__main__":
    main()
