import streamlit as st
import requests
import mlflow


mlflow.set_tracking_uri("http://127.0.0.1:5000")  # URI du serveur MLflow
mlflow.set_experiment("Streamlit Predictions") 

# Titre de l'application
st.title("Prédiction du prix des maisons en Californie")
st.write("Entrez les caractéristiques des maisons pour obtenir une prédiction.")

# Formulaire pour collecter les entrées utilisateur
medinc = st.number_input("Revenu médian (MedInc)", min_value=0.0, step=0.1, format="%.2f")
house_age = st.number_input("Âge médian des maisons (HouseAge)", min_value=0.0, step=1.0, format="%.1f")
ave_rooms = st.number_input("Nombre moyen de pièces (AveRooms)", min_value=0.0, step=0.1, format="%.2f")
ave_bedrms = st.number_input("Nombre moyen de chambres (AveBedrms)", min_value=0.0, step=0.1, format="%.2f")
population = st.number_input("Population", min_value=0, step=1, format="%d")
ave_occup = st.number_input("Occupation moyenne (AveOccup)", min_value=0.0, step=0.1, format="%.2f")
latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, step=0.1, format="%.2f")
longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, step=0.1, format="%.2f")

# Bouton pour envoyer la requête
if st.button("Prédire"):
    # Construire la requête JSON
    input_data = {
        "MedInc": medinc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude,
    }

    # Envoyer la requête à l'API FastAPI
    try:
        response = requests.post("http://127.0.0.1:8000/predict/", json=input_data)
        if response.status_code == 200:
            prediction = response.json().get("prediction", "Erreur")
            st.success(f"Prix prédit : {prediction:.2f} (en $100,000)")
            # Logger les paramètres et la prédiction dans MLflow
            
            with mlflow.start_run(run_name="Streamlit Prediction"):
                mlflow.log_params(input_data)  # Log des paramètres
                mlflow.log_metric("prediction", prediction)  # Log de la prédiction
                mlflow.set_tag("source", "Streamlit UI")  # Tag pour identifier la source

        else:
            st.error(f"Erreur API : {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion à l'API : {e}")
