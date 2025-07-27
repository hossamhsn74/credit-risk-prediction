from pydantic import BaseModel


class UserData(BaseModel):
    """
    User data model for credit risk prediction.

    Attributes:
        person_age (float): Age of the person.
        person_income (float): Annual income of the person.
        person_home_ownership (str): Home ownership status.
        person_emp_length (float): Employment length in years.
        loan_intent (str): Purpose of the loan.
        loan_grade (str): Grade assigned to the loan.
        loan_amnt (float): Amount of the loan.
        loan_int_rate (float): Interest rate of the loan.
        loan_percent_income (float): Loan amount as a percentage of income.
        cb_person_default_on_file (str): Whether the person has defaulted before.
        cb_person_cred_hist_length (float): Length of credit history.
    """
    person_age: float
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: float
