from fastapi.testclient import TestClient
from src.api import app
client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"

def test_predict_rejects_missing_fields():
    response = client.post("/predict", json={"age": 55})
    assert response.status_code == 422
