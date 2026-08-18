# Home Credit Default Risk

## Contexte

Ce projet s'appuie sur le dataset Home Credit Default Risk (Kaggle) et vise à prédire si un client sera capable de rembourser son crédit (TARGET = 0) ou non (TARGET = 1). Le jeu de données présente un fort déséquilibre de classes, les défauts représentant seulement environ 8% des observations.

L'enjeu métier central est le suivant : accorder un crédit à un client qui fera défaut (faux négatif) est estimé dix fois plus coûteux que refuser un crédit à un client solvable (faux positif). Toutes les décisions de modélisation ont été guidées par cette contrainte.

## Structure du projet

kernel1.ipynb : exploration des données, détection et traitement des anomalies, nettoyage, encodage et feature engineering métier

kernel2.ipynb : agrégation des tables secondaires (bureau, previous applications, installments) et enrichissement du jeu d'entraînement

RandomForest.ipynb : premier modèle (Random Forest), mise en place de la métrique métier et du mécanisme d'optimisation du seuil de décision

LR.ipynb : essai avec une régression logistique (non retenu)

MLP.ipynb : essai avec un réseau de neurones MLP (non retenu)

XGBoost.ipynb : essai avec XGBoost (non retenu)

LightGBM.ipynb : modèle final retenu, optimisation par Optuna, interprétabilité SHAP, enregistrement MLflow et serving REST

## Démarche de sélection du modèle

Le projet a suivi une progression itérative. La première étape a consisté à établir une baseline avec une Random Forest, en introduisant dès ce stade la métrique métier et l'optimisation du seuil. Des essais ont ensuite été menés avec une régression logistique, un MLP et XGBoost, mais aucun n'a surpassé LightGBM sur les critères retenus. LightGBM a finalement été sélectionné pour la combinaison de ses performances sur la métrique métier et de sa rapidité d'entraînement, qui a rendu possible l'optimisation bayésienne des hyperparamètres via Optuna dans un temps raisonnable.

Quatre variantes de LightGBM ont été comparées : pondération des classes, rééchantillonnage, seuil optimisé manuellement, et optimisation complète par Optuna. Le modèle 4 (Optuna) est le meilleur obtenu : AUC de 0.783, recall de 0.702 sur la classe défaut, et un ratio vs naïf de 0.603, soit une réduction de 40% du coût total par rapport à un modèle qui refuserait tous les crédits.

## Métrique métier et détermination du seuil

La métrique centrale est le coût métier total, défini comme suit :

    cout = 10 * faux_negatifs + 1 * faux_positifs

Un faux négatif (client défaillant non détecté) est pénalisé dix fois plus qu'un faux positif (client solvable refusé). Cette asymétrie reflète la réalité opérationnelle du scoring de crédit.

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

## Optimisation des performances d'inférence

**Méthodologie : partir des données de monitoring réelles, pas d'une hypothèse en l'air.** `src/profile_inference.py` commence par interroger `monitoring.py` sur les vrais appels enregistrés dans `logs.db`, et récupère un vrai enregistrement `input_json` (pas une valeur inventée) pour profiler sur une requête authentique. Sur du trafic réel généré via l'API : latence moyenne 0,71ms / p95 1,09ms observées - c'est cette baseline réelle, et le fait qu'elle laissait deviner un potentiel d'optimisation vu l'écart avec les temps de predict mesurés par ailleurs, qui a motivé le profiling détaillé ci-dessous, plutôt qu'une intuition non vérifiée.

Un profiling (`uv run python src/profile_inference.py`) a décomposé `/predict` en étapes chronométrées séparément, sur le modèle réel et un enregistrement réel :

| Étape | Moyenne | p95 |
|---|---|---|
| Construction du DataFrame | 2,30ms | 2,46ms |
| Predict via wrapper `mlflow.pyfunc` (ancien chemin) | 5,27ms | 5,86ms |
| Predict LightGBM natif (sans `mlflow.pyfunc`) | 1,27ms | 1,84ms |

Constat fait : le calcul du modèle lui-même ne prend que 1,27ms, mais le wrapper `mlflow.pyfunc` représentait 75,9% du temps de predict - cohérent avec la première mesure sur donnée synthétique (77,5%), ce qui confirme que le résultat n'était pas un artefact du jeu de test fabriqué.

Un profiling outillé plus fin (`uv run python src/profile_cprofile.py`, via `cProfile`) confirme et localise précisément la source de cet overhead : sur 200 appels, le chemin `mlflow.pyfunc` déclenche 20,3 millions d'appels de fonction Python, contre 1,3 million pour le modèle natif (16x moins). Le vrai goulot est la fonction `_enforce_named_col_schema` de mlflow, qui appelle `pandas.Index.union()` une fois par colonne du schéma - soit ~710 appels à cette opération par requête sur ce modèle (710 features), un pattern O(n_colonnes) coûteux pour l'enforcement de schéma sur un modèle aussi large.

* Découverte annexe (non corrigée, documentée) : ce profiling a aussi révélé que LightGBM sanitize en interne les espaces/virgules dans les noms de colonnes au moment de l'entraînement (ex: `"Business Entity Type 2"` devient `"Business_Entity_Type_2"` dans `model.feature_name_`), alors que le schéma enregistré par `mlflow.pyfunc` (via `infer_signature`) garde les noms originaux avec espaces. Les deux représentations diffèrent sur 143 des 710 colonnes (toutes catégorielles one-hot-encodées). Le modèle natif utilisé en production accepte les deux formes sans erreur dans nos tests, mais un appelant qui construirait ses requêtes à partir du schéma documenté par mlflow (noms avec espaces) plutôt que de `feature_name_` pourrait, en théorie, envoyer des noms de colonnes qui ne correspondent pas exactement à ce que le modèle attend en interne. À surveiller si des colonnes catégorielles sont un jour rejetées de façon inattendue.

* Optimisation appliquée : `src/api.py` charge désormais le modèle LightGBM natif (`mlflow.lightgbm.load_model`) au lieu du wrapper `mlflow.pyfunc`: même modèle, mêmes poids donc aucune perte de précision. Mesuré en conditions réelles sur `/predict` (`uv run python src/benchmark_predict.py`, 100 requêtes HTTP, même payload, avant/après) :

| | Moyenne | Médiane | p95 |
|---|---|---|---|
| Avant (wrapper `mlflow.pyfunc`) | 18,68ms | 15,03ms | 73,47ms |
| Après (modèle natif) | 6,42ms | 6,35ms | 6,97ms |

Résultat : env 3x plus rapide en moyenne, et la variance en queue de distribution (p95) est passée de 73ms à 7ms.

Pour reproduire la mesure "avant" : `benchmark_predict.py` mesure `/predict` tel qu'il existe dans le code actuel (donc la version optimisée). Le commit `38a1613` (juste avant cette optimisation) contient encore l'ancien code base sur `mlflow.pyfunc` :

    git checkout 38a1613 -- src/api.py
    # relancer l'API, puis uv run python src/benchmark_predict.py
    git checkout HEAD -- src/api.py

* Deuxième round d'optimisation : le profiling initial montrait que la construction du `pd.DataFrame` (2,42ms) coûtait à elle seule plus que le calcul natif du modèle (1,13ms). En contournant aussi pandas et la couche `LGBMClassifier.predict_proba` - appel direct de `model.booster_.predict()` sur un tableau numpy construit à la main, dans l'ordre exact de `model.feature_name_` - la comparaison isolée montre un gain supplémentaire de ~15x (`predict_proba(df)` : 1,09ms vs `booster_.predict(array)` : 0,07ms, résultats numériquement identiques, vérifié avec `np.allclose`).

Mesuré en conditions réelles sur `/predict` complet :

| | Moyenne | Médiane | p95 |
|---|---|---|---|
| Round 1 (modèle natif via `predict_proba(df)`) | 6,42ms | 6,35ms | 6,97ms |
| Round 2 (`booster_.predict` sur tableau numpy) | 3,67ms | 3,57ms | 4,42ms |

Au total depuis le point de départ : 18,68ms → 3,67ms, soit environ 5x plus rapide!

* Impact sur la précision : aucun. Contrairement à une optimisation type quantification ou distillation, aucune des deux optimisations n'altère le calcul du modèle lui-même : même modèle, mêmes poids, seul le chemin d'accès en amont change. varification concrète:

- Round 1 (bypass `mlflow.pyfunc`) : le modèle natif chargé (`lightgbm-credit-scoring`) et le wrapper pyfunc (`lightgbm-credit-scoring-serving`) proviennent du même run d'entraînement (vérifié via les `run_id` MLflow), donc des mêmes poids.
- Round 2 (bypass pandas/sklearn) : `model.booster_.predict()` appelle directement le même booster que `LGBMClassifier.predict_proba()` en interne, sans transformation intermédiaire des valeurs.
- Validation numérique (`tests/test_no_regression.py`, automatisée en CI) : comparaison des probabilités retournées par l'ancien chemin (`mlflow.pyfunc`) et le nouveau (`booster_.predict`) sur 30 échantillons aux caractéristiques variées / écart maximal observé : 0.0 (tolérance testée : 1e-9). Puisque les probabilités sont identiques, toutes les métriques business qui en dérivent (AUC, recall, coût métier, seuil optimal) le sont aussi mécaniquement donc inutile de les recalculer.

* Optimisations envisagées mais non retenues : réduire la complexité du modèle aurait dégradé le coût métier déjà optimisé (cf. "Démarche de sélection du modèle") pour un gain marginal, vu que le calcul natif est déjà sous 1,5ms : ça a été estimé non justifié.

## Justification de la configuration finale

* Modèle et inférence (software) : LightGBM natif, chargé directement via `mlflow.lightgbm` plutôt que par le wrapper générique `mlflow.pyfunc` (cf. section précédente, gain mesuré de env 3x). Une compilation vers ONNX ou treelite a été envisagée mais écartée : le calcul natif est déjà sous 1,5ms, l'essentiel du gain potentiel (contourner l'overhead pyfunc) est déjà obtenu, et ajouter une étape de compilation/conversion du modèle introduirait de la complexité de déploiement (nouvelle dépendance, pipeline de conversion, risque de divergence numérique) pour un gain marginal à ce niveau de latence déjà faible.

* Hardware : CPU uniquement, pas de GPU : LightGBM est un ensemble d'arbres de décision, pas un réseau de neurones : l'inférence sur CPU est déjà optimale pour ce type de modèle. Avec un modèle à 398 arbres inférant en env 1ms sur un CPU standard, un GPU n'apporterait aucun bénéfice pour ce cas d'usage et ajouterait un coût d'infrastructure et de complexité de déploiement injustifiés.

* API : FastAPI + Uvicorn : Choisi pour la documentation Swagger générée automatiquement, la validation de schéma native (Pydantic), et le support asynchrone - largement suffisant pour ce volume de requêtes, sans le poids d'un framework plus lourd.

* Stockage des logs et du tracking MLflow : SQLite : Zéro administration, suffisant pour le volume d'un PoC local (cf. section "Points de vigilance"). Une base de production à plus grande échelle nécessiterait sans doute PostgreSQL ou équivalent, mais ce serait disproportionné ici.

* Dashboard : Streamlit plutôt que Dash : Développement plus rapide pour un dashboard de monitoring interne aux besoins simples.

## Dashboard de monitoring

    uv run streamlit run src/dashboard.py

Le dashboard (taux d'erreur, latence, dérive des données) est accessible à http://localhost:8501.

## Tests et couverture

Les tests unitaires et d'intégration se lancent avec :

    uv run pytest tests/ --cov --cov-report=term-missing

```
Name                 Stmts   Miss  Cover   Missing
--------------------------------------------------
src/api.py              29      0   100%
src/final_model.py      73     43    41%   34-45, 78-117, 121-144, 148
--------------------------------------------------
TOTAL                  102     43    58%
```

`src/api.py` est couvert à 100% grâce aux tests d'intégration (`tests/test_api.py`) qui exercent chaque route (`/health`, `/predict`) avec un modèle factice injecté, indépendamment du vrai modèle MLflow.

`src/final_model.py` est à 41%, un taux volontairement bas : les fonctions testées unitairement (`cout_metier`, `seuil_optimal`) sont les seules à contenir de la logique métier critique et sont couvertes à 100%. Les 43 lignes non couvertes (`load_data`, `train_and_register`, `_register_serving_wrapper`) correspondent au pipeline d'entraînement et d'enregistrement MLflow - du code d'orchestration qui lit des données lourdes (`data/train_engineered.csv`, ~1,2 Go) et écrit dans le Model Registry. Le tester en continu demanderait soit de le ré-exécuter à chaque run de tests (coûteux en temps et en ressources, et non déterministe; cf. la variance de seuil observée entre deux runs), soit de le mocker intégralement, ce qui ne testerait plus que du câblage et non un vrai comportement. Ce code est donc validé manuellement (voir la section "Démarche de sélection du modèle") plutôt que par des tests automatisés.

Un taux de couverture global à 58% n'est donc pas un signal d'alerte ici : il reflète une répartition volontaire entre code testé unitairement (logique métier pure) et code d'orchestration ML (entraînement, I/O, MLflow) validé autrement.

## Dépendances

Les dépendances sont gérées avec uv. Pour installer l'environnement :

    uv sync
