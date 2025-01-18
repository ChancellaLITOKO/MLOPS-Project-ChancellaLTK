from fastapi.testclient import TestClient
import sys
import os
import mlflow

# Add the src directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.api import app  # Import the FastAPI app

client = TestClient(app)

# Test for the root endpoint
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Bienvenue sur l'API de prédiction du prix des maisons en Californie."}

# Test for the predict endpoint
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

# Test to verify MLflow integration
def test_mlflow_integration():
    # Ensure MLflow is tracking to the expected server
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Start an MLflow run
    with mlflow.start_run():
        run_id = mlflow.active_run().info.run_id

        # Log a metric, parameter, and artifact to the MLflow server
        mlflow.log_param("test_param", "value")
        mlflow.log_metric("test_metric", 1.23)

        # Ensure the run ID is valid
        assert run_id is not None

    # Verify the run exists on the MLflow server
    run_data = mlflow.get_run(run_id)
    assert run_data.info.run_id == run_id
    assert run_data.data.metrics["test_metric"] == 1.23
    assert run_data.data.params["test_param"] == "value"

