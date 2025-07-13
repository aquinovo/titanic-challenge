import os
from fastapi.testclient import TestClient
from api import app
from dotenv import load_dotenv


# Load .env API key
load_dotenv()
API_KEY = os.environ["TITANIC_API_KEY"]

client = TestClient(app)
HEADERS = {"X-API-Key": API_KEY}

def test_predict_rf():
    passengers = [{
        "Age": 22.0, "SibSp": 1, "Parch": 0, "Fare": 7.25,
        "C": False, "Q": False, "S": True,
        "female": False, "male": True,
        "Class_1": False, "Class_2": False, "Class_3": True
    }]
    response = client.post("/predict?model=rf", json=passengers, headers=HEADERS)
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert isinstance(response.json()["predictions"], list)

def test_predict_svc():
    passengers = [{
        "Age": 38.0, "SibSp": 1, "Parch": 0, "Fare": 71.2833,
        "C": True, "Q": False, "S": False,
        "female": True, "male": False,
        "Class_1": True, "Class_2": False, "Class_3": False
    }]
    response = client.post("/predict?model=svc", json=passengers, headers=HEADERS)
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert isinstance(response.json()["predictions"], list)

def test_feature_importance():
    response = client.get("/feature_importance", headers=HEADERS)
    assert response.status_code == 200
    imp = response.json()
    # Simple check: should be a dict with expected keys
    assert isinstance(imp, dict)
    for expected_feature in [
        "Age", "SibSp", "Parch", "Fare", "C", "Q", "S",
        "female", "male", "Class_1", "Class_2", "Class_3"
    ]:
        assert expected_feature in imp
