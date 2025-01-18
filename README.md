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

Pour comprendre l'importance des features sur les predictions nous avons calculer les importances globales des features a l'aide des features importances et egalement fait une analyse de l'impact local pour des exemples indicuduels grace a SHAP. 
Voici quelques resultats : 
```

```



