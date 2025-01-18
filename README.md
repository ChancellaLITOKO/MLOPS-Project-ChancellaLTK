# Projet MLOps pour les étudiants : Prédiction de prix immobiliers en Californie

## Objectif principal
Ce projet vise à introduire les étudiants à la gestion complète du cycle de vie d’un modèle de machine learning, en utilisant des outils modernes et une approche structurée MLOps. Les étapes couvriront la modélisation, la mise en production, et le suivi du modèle en production.

## Scénario
Vous travaillez en tant que Data Scientist/Machine Learning Engineer pour une entreprise immobilière fictive nommée "ImmoPrix". Votre mission est de développer un modèle capable de prédire le prix médian des maisons en Californie pour aider les agents à fixer des prix justes.

Les données sont issues du jeu de données California Housing. Voici les caractéristiques principales :

**MedInc :**  Revenu médian (en 10K $).

**HouseAge :** Âge médian des maisons.

**AveRooms :** Nombre moyen de pièces par logement.

**AveBedrms :** Nombre moyen de chambres par logement.

**Population :** Taille de la population dans le secteur.

**AveOccup :** Occupation moyenne par maison.

**Latitude :** Latitude.

**Longitude :** Longitude.

L’objectif est de prédire MedHouseVal, le prix médian des maisons (en 100K $).

---
## 📊 MISSION 1: Exploration et préparation des données

Cette étape a été réalisée dans un notebook Jupyter et le rapport a été sauvegardé au format PDF. Vous pouvez retrouver ce rapport dans ce dépôt, dans le dossier `.github`, sous le nom **`rapport_exploratoire.pdf`**.

---

## 🤖 MISSION 2: Modélisation et suivi des expériences

Trois modèles de machine learning ont été créés pour cette tâche :
1. **Linear Regression**
2. **Random Forest**
3. **Gradient Boosting**

### 🎯 Meilleur modèle retenu
Après évaluation des performances des modèles, le **Random Forest** a été sélectionné comme meilleur modèle en raison de sa précision supérieure. Ce modèle a été enregistré dans **MLflow** sous le nom **Best Model**.

### 🛠️ Tracking des expériences avec MLflow
Dans **MLflow**, les éléments suivants sont disponibles :
- Les **logs des modèles**, incluant les hyperparamètres, métriques et courbes d'évaluation.
- Les **tests d'API**, permettant de valider les performances de l'API.
- Les **entrées de données** utilisées pour l'application Streamlit.

Vous pouvez consulter l’historique des expérimentations directement dans l’interface MLflow.

---

## MISSION 3 : Analyse des features

L'objectif de cette mission est de comprendre l'importance des différentes features utilisées par le modèle pour effectuer ses prédictions. Pour cela, deux types d'analyses ont été réalisées :

1. **Analyse globale des features** : Identifier les features les plus influentes sur les prédictions grâce à l'attribut `feature_importances_` du modèle Random Forest.
2. **Analyse locale des features** : Évaluer l'impact des features pour des exemples individuels (5 échantillons dans notre cas) à l'aide de SHAP.
   
Voici quelques resultats : 
```
Importances globales des features :
      Feature  Importance
0      MedInc    0.524879
5    AveOccup    0.138447
6    Latitude    0.088952
7   Longitude    0.088630
1    HouseAge    0.054615
2    AveRooms    0.044276
4  Population    0.030609
3   AveBedrms    0.029591
```

```
Analyse des valeurs SHAP pour l'échantillon 1 (index: 949):
   Feature  SHAP Value  Feature Value
  AveOccup    0.861297      -0.104657
  Latitude    0.348161      -0.689483
 Longitude    0.172066       0.599469
  HouseAge    0.107161       0.348490
 AveBedrms    0.100586       0.141917
  AveRooms    0.049043      -0.579824
Population   -0.007533      -0.604608
    MedInc   -0.550359      -0.556630
```

**Conclusion :** L'analyse des features a révélé que 

**- Le revenu médian (MedInc)** est la variable la plus influente au niveau global, ce qui est cohérent avec la corrélation attendue entre le revenu des habitants et la valeur des maisons.

**- Les analyses locales avec SHAP** permettent de comprendre, pour chaque prédiction individuelle, quelles features ont le plus influencé le modèle. Cela est utile pour justifier des prédictions spécifiques ou pour identifier des anomalies.

---

## MISSION 4: Mise en Production
L'objectif est d'exposer le modèle de prédiction du prix des maisons en Californie via une API, afin de rendre le modèle accessible pour des utilisateurs ou des systèmes externes. Cette mission inclut également la création d'une interface utilisateur locale pour tester l'API.

### Étapes Réalisées

#### 1. Création de l'API avec FastAPI
- Une API a été développée à l'aide de **FastAPI**, permettant de :
  - Recevoir des données d'entrée au format JSON (par exemple : revenu médian, âge des maisons, nombre moyen de chambres, etc.).
  - Effectuer une prédiction en utilisant le modèle Random Forest entraîné.
  - Retourner la prédiction sous forme de réponse JSON.

**Exemple de point d'entrée dans l'API** :
- **GET** `/` : Retourne un message de bienvenue.
- **POST** `/predict/` : Accepte des données en entrée et retourne une prédiction.

#### 2. Conteneurisation avec Docker
- L'API a été conteneurisée à l'aide de **Docker** pour garantir sa portabilité et son déploiement sur n'importe quel environnement compatible avec Docker.
- Un fichier `Dockerfile` a été créé pour :
  - Installer les dépendances nécessaires.
  - Exécuter l'API à l'aide d'Uvicorn, un serveur d'applications rapide pour FastAPI.

**Commandes Docker principales** :

- **Construire l'image Docker** :
  ```bash
  docker build -t fastapi-api .
  ```
  Une fois lancé, l'API est accessible localement via : http://127.0.0.1:8000/docs.
  
  #### 3. Interface utilisateur locale avec Streamlit
  Une interface utilisateur a été créée avec Streamlit pour permettre aux utilisateurs de tester l'API facilement.
L'application Streamlit permet d'entrer des données dans un formulaire intuitif, d'envoyer les données à l'API et de visualiser les prédictions renvoyées.

**Commandes pour lancer streamlit**
```
streamlit run app_streamlit.py
```
Une fois lancée, l'interface est accessible via : http://localhost:8501.

---

## Mission 5 : Implémentation des Bonnes Pratiques MLOps

### Étapes Réalisées

#### 1. Configuration de Git pour la gestion du code
- **Git** a été utilisé pour gérer l'ensemble du code source et des fichiers du projet.
- Un dépôt GitHub a été créé pour centraliser les versions et faciliter la collaboration. Lien vers le dépôt : [Lien vers le dépôt GitHub](https://github.com/ChancellaLITOKO/MLOPS-Project-ChancellaLTK).
- Les bonnes pratiques de gestion de version ont été suivies :
  - Création d'un fichier `.gitignore` pour exclure les fichiers inutiles (par exemple, `__pycache__`, fichiers de dépendances locales, etc.).
  - Des commits fréquents et bien documentés ont été réalisés.

#### 2. Mise en place d’un pipeline CI/CD avec GitHub Actions
- Un pipeline CI/CD a été configuré avec **GitHub Actions** pour automatiser les processus suivants :
  - **Tests unitaires** : Utilisation de `pytest` pour exécuter les tests afin de garantir la qualité du code.
  - **Build Docker** : Construction de l'image Docker pour conteneuriser l'API.
  - **Déploiement automatisé** : Push de l'image Docker sur Docker Hub pour un déploiement simplifié.

**Fichier GitHub Actions (`.github/workflows/ci-cd.yml`)** :
```yaml
name: CI/CD Pipeline

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

jobs:
  build:
    runs-on: ubuntu-22.04

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.12.8'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install poetry
        poetry install


    - name: Run Tests
      run: |
        export DISABLE_MLFLOW=True
        poetry run pytest tests/


  deploy:
    needs: build
    runs-on: ubuntu-22.04

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Build Docker Image
      run: docker build -t fastapi-api .

    - name: Push Docker Image to Docker Hub
      env:
        DOCKER_USERNAME: ${{ secrets.DOCKER_USERNAME }}
        DOCKER_PASSWORD: ${{ secrets.DOCKER_PASSWORD }}
      run: |
        echo $DOCKER_PASSWORD | docker login -u $DOCKER_USERNAME --password-stdin
        docker tag fastapi-api:latest $DOCKER_USERNAME/fastapi-api:latest
        docker push $DOCKER_USERNAME/fastapi-api:latest
```



