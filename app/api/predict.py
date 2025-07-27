from fastapi import APIRouter
from app.models.user_data import UserData
import pandas as pd
from ml.predictor import feature_engineering, preprocess_data, load_artifacts

# Create a router for prediction endpoints
router = APIRouter()

# Load model, encoder, and scaler at startup (singleton pattern)
model, ohe, scaler = load_artifacts(prefix="catboost")


@router.post("/predict")
def predict(user: UserData):
    """
    Predict the risk score for a new user.

    Args:
        user (UserData): User input data.

    Returns:
        dict: Dictionary with the predicted risk score.
    """
    user_df = pd.DataFrame([user.dict()])
    user_df = feature_engineering(user_df)
    user_X, _, _ = preprocess_data(user_df, ohe=ohe, scaler=scaler, fit=False)
    prob = model.predict_proba(user_X)[0][1]
    percent_score = f"{prob * 100:.2f}%"
    return {"risk_score": percent_score}
