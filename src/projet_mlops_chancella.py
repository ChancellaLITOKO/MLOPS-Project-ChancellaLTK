
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

def load_data():
    """
    Charger les données California Housing et ajouter la colonne cible 'MedHouseVal'.
    """
    housing = fetch_california_housing(as_frame=True)
    housing.data["MedHouseVal"] = housing.target
    return housing.data

def explore_data(data):
    """
     afficher les informations principales et visualiser la distribution de la cible.
    """
    # Aperçu des données
    print(data.info())
    print("\nDescription statistique des données :")
    print(data.describe())

    # Vérifier les valeurs manquantes
    print("\nValeurs manquantes :")
    print(data.isnull().sum())

    # Visualiser la distribution de la variable cible
    plt.figure(figsize=(10, 6))
    sns.histplot(data["MedHouseVal"], bins=50, kde=True, color='blue')
    plt.title("Distribution des prix des maisons (MedHouseVal)")
    plt.xlabel("Prix médian des maisons (en 100k $)")
    plt.ylabel("Fréquence")
    plt.show()

def prepare_data(data):
    """
    Préparer les données : diviser en ensembles d'entraînement/test et appliquer une standardisation.
    """
    # Séparer les features (X) et la cible (y)
    X = data.drop("MedHouseVal", axis=1)
    y = data["MedHouseVal"]

    # Diviser les données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardisation des données
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Vérification des tailles des ensembles
    print("Taille des ensembles après division :")
    print(f"X_train : {X_train.shape}, X_test : {X_test.shape}")
    print(f"y_train : {y_train.shape}, y_test : {y_test.shape}")

    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    # Charger les données
    data = load_data()
    explore_data(data)
    X_train_scaled, X_test_scaled, y_train, y_test = prepare_data(data)

