from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field


class CensusInput(BaseModel):
    age: int
    workclass: Literal[
        'State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov', 'Local-gov',
        'Self-emp-inc', 'Without-pay', 'Never-worked'
    ]
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: Literal['Male', 'Female']
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    class Example:
        schema = {
                "age": 33,
                "workclass": "State-gov",
                "fnlgt": 160187,
                "capital-gain": 2174,
                "education": "Bachelors",
                "marital-status": "Never-married",
                "hours-per-week": 40,
                "education-num": 13,
                "occupation": "Adm-clerical",
                "race": "Asian-Pac-Islander",
                "relationship": "Not-in-family",
                "sex": "Male",
                "capital-loss": 0,
                "native-country": "United-States"
        }


def body_to_df(input_data: CensusInput):
    input_df = pd.DataFrame([input_data.model_dump()])
    column_mapping = {
        "education_num": "education-num",
        "marital_status": "marital-status",
        "capital_gain": "capital-gain",
        "capital_loss": "capital-loss",
        "hours_per_week": "hours-per-week",
        "native_country": "native-country"
    }
    input_df.rename(columns=column_mapping, inplace=True)

    return input_df
