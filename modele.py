import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from projet_mlops_chancella import load_data, prepare_data # Import des fonctions existantes

# Charger et préparer les données
data = load_data()
X_train_scaled, X_test_scaled, y_train, y_test = prepare_data(data)

def train_and_log_linear_regression(X_train, X_test, y_train, y_test):
    """
    Entraîner un modèle de régression linéaire, calculer les métriques, et enregistrer les résultats avec MLflow.
    """
    # Définir le modèle
    model_lr = LinearRegression()

    with mlflow.start_run(run_name="LinearRegression"):
        # Entraîner le modèle
        model_lr.fit(X_train, y_train)
        y_pred =  model_lr.predict(X_test)

        # Calculer les métriques
        rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred))
        r2_lr = r2_score(y_test, y_pred)

        # Enregistrer les métriques dans MLflow
        mlflow.log_metric("rmse", rmse_lr)
        mlflow.log_metric("r2", r2_lr)

        # Enregistrer le modèle dans MLflow
        mlflow.sklearn.log_model(model_lr, "LinearRegression")

        print(f"Linear Regression - RMSE : {rmse_lr:.2f}, R² : {r2_lr:.2f}")

    return model_lr, rmse_lr

def train_random_forest(X_train, X_test, y_train, y_test):
    """
    Entraîner un modèle Random Forest et enregistrer les résultats avec MLflow.
    """
    model_name = "RandomForest"
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    
    with mlflow.start_run(run_name=model_name):
        # Entraîner le modèle
        model_rf.fit(X_train, y_train)
        y_pred = model_rf.predict(X_test)

        # Calcul des métriques
        rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred))
        mae_rf= mean_absolute_error(y_test, y_pred)
        r2_rf= r2_score(y_test, y_pred)

        # Enregistrer les métriques et le modèle
        mlflow.log_metric("rmse", rmse_rf)
        mlflow.log_metric("mae", mae_rf)
        mlflow.log_metric("r2", r2_rf)
        
        mlflow.sklearn.log_model(model_rf, model_name)

        print(f"{model_name} - RMSE: {rmse_rf:.2f}, MAE: {mae_rf:.2f}, R²: {r2_rf:.2f}")

    return model_rf, rmse_rf

def train_gradient_boosting(X_train, X_test, y_train, y_test):
    """
    Entraîner un modèle Gradient Boosting et enregistrer les résultats avec MLflow.
    """
    model_name = "GradientBoosting"
    model_gb= GradientBoostingRegressor(n_estimators=100, random_state=42)
    
    with mlflow.start_run(run_name=model_name):
        # Entraîner le modèle
        model_gb.fit(X_train, y_train)
        y_pred = model_gb.predict(X_test)

        # Calcul des métriques
        rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred))
        mae_gb = mean_absolute_error(y_test, y_pred)
        r2_gb = r2_score(y_test, y_pred)

        # Enregistrer les métriques et le modèle
        mlflow.log_metric("rmse", rmse_gb)
        mlflow.log_metric("mae", mae_gb)
        mlflow.log_metric("r2", r2_gb)
        mlflow.sklearn.log_model(model_gb, model_name)

        print(f"{model_name} - RMSE: {rmse_gb:.2f}, MAE: {mae_gb:.2f}, R²: {r2_gb:.2f}")

    return model_gb, rmse_gb

if __name__ == "__main__":
    # Configurer MLflow
    mlflow.set_experiment("Projet_MLOps_Chancella")   # Définir un nom d'expérience

    # Entraîner les modèles et comparer les performances
    results = []

    # Entraîner chaque modèle et enregistrer les résultats
    print("\nEntraînement du modèle Linear Regression...")
    model_lr, rmse_lr = train_and_log_linear_regression(X_train_scaled, X_test_scaled, y_train, y_test)
    results.append(("LinearRegression", model_lr, rmse_lr))

    print("\nEntraînement du modèle Random Forest...")
    model_rf, rmse_rf = train_random_forest(X_train_scaled, X_test_scaled, y_train, y_test)
    results.append(("RandomForest", model_rf, rmse_rf))

    print("\nEntraînement du modèle Gradient Boosting...")
    model_gb, rmse_gb = train_gradient_boosting(X_train_scaled, X_test_scaled, y_train, y_test)
    results.append(("GradientBoosting", model_gb, rmse_gb))

    # Le meilleur modèle (celui avec le RMSE le plus faible)
    best_model_name, best_model, best_rmse = min(results, key=lambda x: x[2])
    print(f"\nLe meilleur modèle est {best_model_name} avec un RMSE de {best_rmse:.2f}")





