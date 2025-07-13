from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_predict_rf():
    passengers = [{
        "Age": 22.0, "SibSp": 1, "Parch": 0, "Fare": 7.25,
        "C": False, "Q": False, "S": True,
        "female": False, "male": True,
        "Class_1": False, "Class_2": False, "Class_3": True
    }]
    response = client.post("/predict?model=rf", json=passengers)
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
    response = client.post("/predict?model=svc", json=passengers)
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert isinstance(response.json()["predictions"], list)
