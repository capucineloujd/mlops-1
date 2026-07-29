# Home Credit Default Risk

## Contexte

Ce projet s'appuie sur le dataset Home Credit Default Risk (Kaggle) et vise à prédire si un client sera capable de rembourser son crédit (TARGET = 0) ou non (TARGET = 1). Le jeu de données présente un fort déséquilibre de classes, les défauts représentant environ 8% des observations.

L'enjeu métier central est asymétrique : accorder un crédit à un client qui fera défaut (faux négatif) est estimé dix fois plus coûteux que refuser un crédit à un client solvable (faux positif). Toutes les décisions de modélisation ont été guidées par cette contrainte.

## Structure du projet

kernel1.ipynb : exploration des données, détection et traitement des anomalies, nettoyage, encodage et feature engineering métier

kernel2.ipynb : agrégation des tables secondaires (bureau, previous applications, installments) et enrichissement du jeu d'entraînement

model.ipynb : premier modèle (Random Forest), mise en place de la métrique métier et du mécanisme d'optimisation du seuil de décision

LR.ipynb : essai avec une régression logistique (non retenu)

MLP.ipynb : essai avec un réseau de neurones MLP (non retenu)

xgboost.ipynb : essai avec XGBoost (non retenu)

lightGBM.ipynb : modèle final retenu, optimisation par Optuna, interprétabilité SHAP, enregistrement MLflow et serving REST

## Démarche de sélection du modèle

Le projet a suivi une progression itérative. La première étape a consisté à établir une baseline avec une Random Forest, en introduisant dès ce stade la métrique métier et l'optimisation du seuil. Des essais ont ensuite été menés avec une régression logistique, un MLP et XGBoost, mais aucun n'a surpassé LightGBM sur les critères retenus. LightGBM a finalement été sélectionné pour la combinaison de ses performances sur la métrique métier et de sa rapidité d'entraînement, qui a rendu possible l'optimisation bayésienne des hyperparamètres via Optuna dans un temps raisonnable.

Quatre variantes de LightGBM ont été comparées : pondération des classes, rééchantillonnage, seuil optimisé manuellement, et optimisation complète par Optuna. Le modèle 4 (Optuna) est le meilleur obtenu : AUC de 0.783, recall de 0.702 sur la classe défaut, et un ratio vs naïf de 0.603, soit une réduction de 40% du coût total par rapport à un modèle qui refuserait tous les crédits.

## Métrique métier et détermination du seuil

La métrique centrale est le coût métier total, défini comme suit :

    cout = 10 * faux_negatifs + 1 * faux_positifs

Un faux négatif (client défaillant non détecté, prêt accordé à tort) est pénalisé dix fois plus qu'un faux positif (client solvable refusé). Cette asymétrie reflète la réalité opérationnelle du scoring de crédit.

Le seuil de décision n'est pas fixé arbitrairement à 0.5. Pour chaque modèle, l'ensemble des seuils possibles issus de la courbe ROC est parcouru, et le seuil minimisant le coût métier sur le jeu de validation est retenu. Pour le modèle final, ce seuil est de 0.499. Cette méthode garantit que la règle de décision est alignée sur l'objectif économique et non sur une convention statistique.

En complément, un ratio vs naïf est calculé : il compare le coût du modèle à celui d'un classifieur trivial qui refuserait l'intégralité des crédits. Ce ratio permet d'évaluer la valeur ajoutée réelle du modèle indépendamment de l'échelle du dataset.

## Rôle de MLflow

MLflow est utilisé tout au long du projet comme outil de traçabilité et de gestion du cycle de vie des modèles.

Chaque entraînement fait l'objet d'un run MLflow qui enregistre automatiquement les hyperparamètres, les métriques (AUC, recall, précision, coût métier, ratio vs naïf), les courbes ROC et les importances de variables. Cela permet de comparer objectivement toutes les expériences dans une interface unifiée, sans risque de perte ou de confusion entre les résultats.

Les modèles sont versionnés dans le Model Registry MLflow sous le nom `lightgbm-credit-scoring`. Le meilleur modèle reçoit l'alias `gagnant`, ce qui permet de l'identifier sans dépendre d'un numéro de version arbitraire.

Une version pyfunc du modèle final est également enregistrée sous `lightgbm-credit-scoring-serving`. Ce wrapper retourne des probabilités de défaut (et non des classes binaires) et peut être servi directement via l'API REST de MLflow (`mlflow models serve`), rendant le modèle consommable par n'importe quel client HTTP sans dépendance à LightGBM.

## Lancer l'interface MLflow

    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

L'interface est accessible à http://localhost:5000.

## Dépendances

Les dépendances sont gérées avec uv. Pour installer l'environnement :

    uv sync
