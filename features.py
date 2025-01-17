import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from projet_mlops_chancella import load_data, prepare_data  # Import des fonctions existantes
from modele import train_random_forest  # Importer la fonction pour entraîner le modèle Random Forest

# Charger et préparer les données
data = load_data()
X_train_scaled, X_test_scaled, y_train, y_test = prepare_data(data)
feature_names = data.columns[:-1]  # Noms des features (toutes sauf la cible)

# Convertir X_train_scaled et X_test_scaled en DataFrame avec noms des colonnes
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)

# Entraîner le modèle Random Forest et récupérer l'objet du modèle
print("\nEntraînement du modèle Random Forest...")
model_rf, _ = train_random_forest(X_train_scaled, X_test_scaled, y_train, y_test)

# Analyse globale des features avec feature_importances_
def analyze_global_importance(model, feature_names):
    """
    Analyse globale des importances des features avec l'attribut feature_importances_.
    """
    feature_importances = model.feature_importances_

    # Création d'un DataFrame pour les importances
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    # Afficher les importances dans le terminal
    print("\nImportances globales des features :")
    print(importance_df)

    # Visualiser les importances globales
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Importances globales des features (Random Forest)")
    plt.gca().invert_yaxis()
    plt.show()

# Analyse locale des valeurs SHAP
def analyze_local_importance(model, X_sample, feature_names):
    """
    Analyse locale des valeurs SHAP pour des exemples individuels.
    """
    print("\n[INFO] Calcul des valeurs SHAP locales...")

    # Vérification des dimensions et du type de X_sample
    if not isinstance(X_sample, pd.DataFrame):
        raise ValueError("X_sample doit être un DataFrame Pandas avec les noms des colonnes correspondants.")

    if X_sample.shape[1] != len(feature_names):
        raise ValueError("Le nombre de colonnes dans X_sample ne correspond pas aux features utilisées pour entraîner le modèle.")

    # Créer un explainer SHAP pour le modèle
    explainer = shap.TreeExplainer(model)
    
    try:
        # Calculer les valeurs SHAP pour les échantillons donnés
        shap_values = explainer.shap_values(X_sample)
        
        # Parcourir chaque échantillon et afficher les valeurs SHAP
        for i, index in enumerate(X_sample.index):
            sample_data = X_sample.loc[index]
            shap_contributions = pd.DataFrame({
                "Feature": feature_names,
                "SHAP Value": shap_values[i],
                "Feature Value": sample_data.values
            })
            print(f"\nAnalyse des valeurs SHAP pour l'échantillon {i + 1} (index: {index}):")
            print(shap_contributions.sort_values(by="SHAP Value", ascending=False).to_string(index=False))
    
    except Exception as e:
        print(f"[ERROR] Une erreur s'est produite lors de l'analyse locale avec SHAP : {e}")

if __name__ == "__main__":
    # Analyse globale des features
    print("\nAnalyse globale des features avec feature_importances_ :")
    analyze_global_importance(model_rf, feature_names)

    # Analyse locale des features pour un sous-ensemble de données
    try:
        X_sample = X_test_scaled.sample(5, random_state=42)  # Sélection de 5 échantillons
        analyze_local_importance(model_rf, X_sample, feature_names)
    except Exception as e:
        print(f"[ERROR] Une erreur s'est produite dans l'analyse SHAP locale : {e}")

""""
L'analyse des features réalisée avec le modèle Random Forest révèle que :

L'analyse globale des features montre que la variable MedInc est la variable la plus influente (plus de 52%), confirmant son rôle central dans la prédiction du prix des maisons en Californie.
AveOccup, ainsi que les coordonnées géographiques Latitude et Longitude, contribuent significativement aux variations des prédictions, soulignant l'importance de la localisation et des dynamiques démographiques.
Les variables comme HouseAge, AveRooms, et Population jouent un rôle secondaire, mais apportent des informations complémentaires.
L'analyse locale à l'aide de SHAP met en évidence que l'impact des features varie selon les observations individuelles. 
Ces variations reflètent des spécificités locales, telles que des disparités régionales ou des comportements atypiques dans les données.
"""




