from http import HTTPStatus

from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import numpy as np
from starter.main import app

client = TestClient(app)

sample_input_data = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}


def given_encoder():
    mock_encoder = MagicMock()
    mock_encoder.transform.return_value = np.random.rand(1, 100)
    return mock_encoder


def test_health_check_root():
    response = client.get("/")

    assert response.status_code == HTTPStatus.OK, (f"Expected {HTTPStatus.OK}"
                                                   f", but got {response.status_code}")

    expected_content = {"status": "Ready"}
    assert response.json() == expected_content, (f"Expected {expected_content},"
                                                 f" but got {response.json()}")


def test_predict_less_than_50k():
    mock_model = MagicMock()
    mock_lb = MagicMock()

    mock_model.predict.return_value = np.array([0])
    mock_lb.inverse_transform.return_value = np.array(['<=50K'])

    client.app.state.model = mock_model
    client.app.state.encoder = given_encoder()
    client.app.state.lb = mock_lb

    response = client.post("/infer", json=sample_input_data)

    assert response.status_code == HTTPStatus.OK
    expected_response = {"predicted_salary": "<=50K"}
    assert response.json() == expected_response


def test_predict_greater_than_50k():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_lb = MagicMock()
    mock_lb.inverse_transform.return_value = np.array(['>50K'])

    client.app.state.model = mock_model
    client.app.state.encoder = given_encoder()
    client.app.state.lb = mock_lb

    response = client.post("/infer", json=sample_input_data)

    assert response.status_code == HTTPStatus.OK
    expected_response = {"predicted_salary": ">50K"}
    assert response.json() == expected_response
