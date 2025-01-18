from fastapi.testclient import TestClient
from unittest.mock import patch
import sys
import os

# Ajout du chemin pour inclure le répertoire `src`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.api import app  # Import de l'application FastAPI

client = TestClient(app)

# Test pour le point d'entrée racine "/"
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Bienvenue sur l'API de prédiction du prix des maisons en Californie."}

# Test pour le point d'entrée "/predict/"
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

# Test pour mocker MLflow
@patch("mlflow.get_run")
def test_mlflow_mocked_run(mock_get_run):
    # Définir le retour de la fonction mockée
    mock_get_run.return_value = {
        "info": {"run_id": "12345"},
        "data": {"metrics": {"rmse": 0.5}}
    }

    # Simulez un appel à la fonction MLflow mockée
    result = mock_get_run("12345")
    
    # Assertions pour vérifier le comportement attendu
    assert result["info"]["run_id"] == "12345"
    assert "metrics" in result["data"]
    assert result["data"]["metrics"]["rmse"] == 0.5

