from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import numpy as np
import os

# Vérifiez si MLflow doit être désactivé
DISABLE_MLFLOW = os.getenv("DISABLE_MLFLOW", "False").lower() == "true"

if not DISABLE_MLFLOW:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    model_uri = "runs:/12aa7debbc504cef8cd65a6625e7dc89/RandomForest"
    model = mlflow.pyfunc.load_model(model_uri)
    mlflow.set_experiment("House Price Prediction Experiment")
else:
    model = None  # Si MLflow est désactivé, ne chargez pas le modèle


# Définir la structure des données d'entrée
class PredictionInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

# Initialiser l'application FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de prédiction du prix des maisons en Californie."}

@app.post("/predict/")
def predict(input_data: PredictionInput):
    try:
        if DISABLE_MLFLOW:
            # Mode sans MLflow : simulation de prédiction
            prediction = 0.0  # Valeur par défaut pour les tests
        else:
            # Convertir les données en tableau NumPy
            data = np.array([[input_data.MedInc, input_data.HouseAge, input_data.AveRooms, 
                              input_data.AveBedrms, input_data.Population, input_data.AveOccup, 
                              input_data.Latitude, input_data.Longitude]])

            # Faire une prédiction
            prediction = model.predict(data)[0]

            # Enregistrer dans MLflow avec un nom de run
            with mlflow.start_run(run_name="API Prediction Run"):
                mlflow.log_param("MedInc", input_data.MedInc)
                mlflow.log_param("HouseAge", input_data.HouseAge)
                mlflow.log_param("AveRooms", input_data.AveRooms)
                mlflow.log_param("AveBedrms", input_data.AveBedrms)
                mlflow.log_param("Population", input_data.Population)
                mlflow.log_param("AveOccup", input_data.AveOccup)
                mlflow.log_param("Latitude", input_data.Latitude)
                mlflow.log_param("Longitude", input_data.Longitude)
                mlflow.log_metric("Prediction", prediction)

        # Retourner le résultat
        return {"prediction": prediction}

    except Exception as e:
        # Loguer l'erreur pour debug
        print(f"Erreur dans /predict : {e}")
        return {"error": f"Erreur interne : {str(e)}"}
