from app.models.user_data import UserData


def generate_reasoning(user: UserData, risk_prob: float) -> tuple[str, list[str]]:
    """
    Generate risk level and reasoning based on user input and risk probability.

    Args:
        user (UserData): Input data from the user.
        risk_prob (float): Predicted risk probability (0 to 1).

    Returns:
        tuple: (risk_level, reasoning list)
    """
    # Risk level based on probability
    if risk_prob < 0.2:
        risk_level = "Low"
    elif risk_prob < 0.5:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    # Reasoning based on user features
    reasoning = []
    if user.person_income < 30000:
        reasoning.append("Low income")
    if user.loan_percent_income > 0.4:
        reasoning.append("High loan-to-income ratio")
    if user.loan_grade in ["F", "G"]:
        reasoning.append("Poor loan grade")
    if user.cb_person_default_on_file == "Y":
        reasoning.append("Previous default history")
    if user.cb_person_cred_hist_length < 12:
        reasoning.append("Short credit history")
    if user.loan_int_rate > 0.2:
        reasoning.append("High interest rate")
    
    if not reasoning:
        reasoning.append("Good financial and credit profile")

    return risk_level, reasoning
