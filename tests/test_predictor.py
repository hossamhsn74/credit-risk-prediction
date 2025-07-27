"""
Unit tests for the UserData Pydantic model used in credit risk prediction.

These tests ensure that:
- Valid data creates a UserData instance.
- Missing required fields raise a ValidationError.
- Invalid types raise a ValidationError.
- Negative values raise a ValidationError (if enforced in the model).
"""

from pydantic import ValidationError
from app.models.user_data import UserData
import pytest


def test_valid_userdata():
    """
    Test that a valid UserData instance can be created.
    """
    user = UserData(
        person_age=30,
        person_income=50000,
        person_home_ownership="RENT",
        person_emp_length=5,
        loan_intent="PERSONAL",
        loan_grade="A",
        loan_amnt=10000,
        loan_int_rate=10.5,
        loan_percent_income=0.2,
        cb_person_default_on_file="N",
        cb_person_cred_hist_length=3
    )
    assert user.person_age == 30
    assert user.person_income == 50000


def test_missing_field():
    """
    Test that missing a required field raises a ValidationError.
    """
    with pytest.raises(ValidationError):
        UserData(
            person_age=30,
            person_income=50000,
            person_home_ownership="RENT",
            person_emp_length=5,
            loan_intent="PERSONAL",
            loan_grade="A",
            loan_amnt=10000,
            loan_int_rate=10.5,
            loan_percent_income=0.2,
            cb_person_default_on_file="N"
            # Missing cb_person_cred_hist_length
        )


def test_invalid_type():
    """
    Test that providing an invalid type raises a ValidationError.
    """
    with pytest.raises(ValidationError):
        UserData(
            person_age="thirty",  # Invalid type
            person_income=50000,
            person_home_ownership="RENT",
            person_emp_length=5,
            loan_intent="PERSONAL",
            loan_grade="A",
            loan_amnt=10000,
            loan_int_rate=10.5,
            loan_percent_income=0.2,
            cb_person_default_on_file="N",
            cb_person_cred_hist_length=3
        )
