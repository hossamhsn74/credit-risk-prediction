# Credit Risk Prediction API

A production-ready FastAPI application for predicting credit risk using machine learning. The API receives applicant data and returns a risk score (probability of default) as a percentage.

---

## Technologies & Tools Used

- **FastAPI**: High-performance web API framework
- **Pydantic**: Data validation and parsing
- **scikit-learn**: Preprocessing, metrics, and KNN model
- **CatBoost, XGBoost, LightGBM**: Gradient boosting ML models
- **pandas**: Data manipulation
- **joblib**: Model serialization
- **pytest**: Unit testing
- **Uvicorn**: ASGI server for FastAPI
- **VS Code Dev Containers**: (optional) Consistent development environment

---

## Project Structure
<pre><code>
credit-risk-prediction/ 
│ ├── app/ 
│ ├─── main.py # FastAPI entrypoint 
│ ├─── api/ 
│ └───── predict.py # Prediction endpoint 
│ └─── models/ 
│ └───── user_data.py # Pydantic model forinput 
│ ├── ml/ 
│ └──── predictor.py # ML pipeline and utilities 
│ ├── model_artifacts/ # Saved model, encoder, scaler 
│ ├──── catboost_model.pkl 
│ ├──── catboost_ohe.pkl 
│ └──── catboost_scaler.pkl 
│ ├── data/ 
│ └──── credit_risk_dataset.csv 
│ ├── tests/ 
│ └──── test_predictor.py # Unit tests 
│── requirements.txt 
├── README.md 
└── .vscode/ 
└── launch.json # Debug config for FastAPI
</code></pre>
---

## Setup Instructions

### 1. Prerequisites

- Python 3.9+
- (Optional) Docker for Dev Containers
- Recommended VS Code extensions:
  - Python
  - Pylance
  - Dev Containers

### 2. Install Dependencies

```sh
pip install -r requirements.txt
```

3. [Optional] Train and Save the Model
Before running the API, train and save the model artifacts:

```sh
python predictor.py
```
This will create files in model_artifacts/ used by the API.

4. Run the FastAPI App
```sh
uvicorn app.main:app --reload --port 5000
```
Or use the VS Code debugger (F5) with the provided .vscode/launch.json.

5. Run Unit Tests
```sh
pytest
```
### 3. API Usage
Endpoint
POST /predict

Example Input

✅ Low Risk Example
```json
{
  "person_age": 45,
  "person_income": 120000,
  "person_home_ownership": "MORTGAGE",
  "person_emp_length": 15,
  "loan_intent": "DEBTCONSOLIDATION",
  "loan_grade": "A",
  "loan_amnt": 10000,
  "loan_int_rate": 0.06,
  "loan_percent_income": 0.08,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 30
}
```

✅ Moderate Risk Example
```json
{
  "person_age": 31,
  "person_income": 66000,
  "person_home_ownership": "RENT",
  "person_emp_length": 6,
  "loan_intent": "PERSONAL",
  "loan_grade": "C",
  "loan_amnt": 15000,
  "loan_int_rate": 0.15,
  "loan_percent_income": 0.23,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 24
}
```

✅ High Risk Example
```json
{
  "person_age": 22,
  "person_income": 28000,
  "person_home_ownership": "RENT",
  "person_emp_length": 0,
  "loan_intent": "VENTURE",
  "loan_grade": "G",
  "loan_amnt": 20000,
  "loan_int_rate": 0.29,
  "loan_percent_income": 0.71,
  "cb_person_default_on_file": "Y",
  "cb_person_cred_hist_length": 6
}

```
Example Output

```json
{
    "risk_score": "99.05%",
    "risk_level": "High",
    "reasoning": [
        "Low income",
        "High loan-to-income ratio",
        "Poor loan grade",
        "Previous default history",
        "Short credit history",
        "High interest rate"
    ]
}
```
The risk_score is the predicted probability (as a percent with two digits on both sides) that the applicant will default.


### 4. API Documentation
Interactive docs available at: http://localhost:5000/docs


### 5. Notes
Retrain the model (python ml/predictor.py) whenever you have new data or want to improve predictions.
The API loads the latest saved model and preprocessing artifacts at startup.
For development, use the provided VS Code Dev Container for a consistent environment.