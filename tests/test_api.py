from fastapi.testclient import TestClient

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src.api import app  

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Bienvenue sur l'API de prédiction du prix des maisons en Californie."}

def test_predict():
    payload = {
        "MedInc": 5.5,
        "HouseAge": 20,
        "AveRooms": 6.0,
        "AveBedrms": 1.0,
        "Population": 300,
        "AveOccup": 3.5,
        "Latitude": 34.0,
        "Longitude": -118.0
    }
    response = client.post("/predict/", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()

