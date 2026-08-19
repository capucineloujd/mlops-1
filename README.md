# Home Credit Default Risk

## Contexte

Ce projet s'appuie sur le dataset Home Credit Default Risk (Kaggle) et vise à prédire si un client sera capable de rembourser son crédit (TARGET = 0) ou non (TARGET = 1). Le jeu de données présente un fort déséquilibre de classes (les défauts représentent environ 8% des observations).

L'enjeu métier est le suivant : accorder un crédit à un client qui fera défaut (faux négatif) est estimé 10x plus coûteux que de refuser un crédit à un client solvable (faux positif).

## Structure du projet

kernel1.ipynb : exploration des données, détection et traitement des anomalies, nettoyage, encodage et feature engineering métier

kernel2.ipynb : agrégation des tables secondaires (bureau, previous applications, installments...) et enrichissement du jeu d'entraînement

RandomForest.ipynb : premier modèle non retenu + mise en place de la métrique métier et du mécanisme d'optimisation du seuil de décision

LR.ipynb : essai avec une régression logistique (non retenu)

MLP.ipynb : essai avec un réseau de neurones MLP (non retenu)

XGBoost.ipynb : essai avec XGBoost (non retenu)

**LightGBM.ipynb** : modèle final retenu optimisé par Optuna + interprétabilité SHAP, enregistrement MLflow et serving REST

## Sélection du modèle

Le projet a suivi une progression itérative. La première étape a consisté à établir une baseline avec une Random Forest, en introduisant la métrique métier et l'optimisation du seuil. Des essais ont ensuite été menés avec une régression logistique, un MLP et XGBoost, mais aucun n'a surpassé LightGBM sur les critères retenus. LightGBM a donc été sélectionné pour la combinaison de ses performances sur la métrique métier et de sa rapidité d'entraînement, qui a permis une optimisation bayésienne des hyperparamètres via Optuna dans un temps raisonnable.

Quatre variantes de LightGBM ont été comparées : pondération des classes, rééchantillonnage, seuil optimisé manuellement, et optimisation complète par Optuna. Le modèle 4 (modèle de base optimisé avec Optuna) est le meilleur obtenu : AUC de 0.783, recall de 0.702 sur la classe défaut, et un ratio vs naïf de 0.603, soit une réduction de 40% du coût total par rapport à un modèle qui refuserait tous les crédits.

## Métrique métier et seuil

La métrique centrale est le coût métier total, défini comme suit :

    cout = 10 * faux_negatifs + 1 * faux_positifs

Un faux négatif (client défaillant non détecté) est pénalisé 10x plus qu'un faux positif (client solvable refusé).

Le seuil de décision n'est pas laissé arbitrairement à 0.5. Pour chaque modèle, l'ensemble des seuils possibles issus de la courbe ROC est parcouru, et le seuil minimisant le coût métier sur le jeu de validation est retenu. Pour le modèle final, ce seuil est de 0.499. Cela permet à ce que la règle de décision soit alignée sur l'objectif économique.

En complément, un ratio vs naïf est calculé : il compare le coût du modèle à celui d'un classifieur trivial qui refuserait l'intégralité des crédits. Ce ratio permet d'évaluer la valeur ajoutée réelle du modèle indépendamment de l'échelle du dataset.

## Rôle de MLflow

MLflow est utilisé tout au long du projet comme outil de traçabilité et de gestion du cycle de vie des modèles.

Chaque entraînement fait l'objet d'un run MLflow qui enregistre les hyperparamètres, les métriques (AUC, recall, précision, coût métier, ratio vs naïf), les courbes ROC et les importances de variables. Cela permet de comparer toutes les expériences.

Les modèles sont versionnés dans le Model Registry MLflow sous le nom `lightgbm-credit-scoring`. Le meilleur modèle reçoit l'alias `gagnant`.

Une version pyfunc du modèle final est également enregistrée sous `lightgbm-credit-scoring-serving`. Ce wrapper retourne des probabilités de défault (et non des classes binaires) et peut être servi directement via l'API REST de MLflow (`mlflow models serve`), rendant le modèle consommable par n'importe quel client HTTP et sans dépendance à LightGBM.

## Lancer l'interface MLflow

    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

L'interface est accessible à http://localhost:5000.

## Stockage des données de production

Chaque appel à `/predict` (succès & échec) est enregistré par `src/storage.py` dans une base PostgreSQL (`DATABASE_URL`), table `prediction_logs` :

| Colonne | Contenu |
|---|---|
| `id`, `timestamp` | identifiant, horodatage UTC |
| `input_json` | les records reçus, en JSON brut |
| `n_records` | nombre de clients dans l'appel |
| `output_json` | probabilités + décisions renvoyées (`null` si l'appel a échoué) |
| `latency_ms` | temps d'exécution mesuré côté serveur |
| `status` | `success` / `error` |
| `error_detail` | message d'erreur (`null` si succès) |

En local/CI, Postgres tourne dans un conteneur dédié (`docker-compose.yml`, service `postgres`, ou service Postgres du job `pytest` en CI). Le stockage est 'best-effort' côté API : une base momentanément injoignable ne fait échouer ni le démarrage (`/health` reste joignable) ni une prédiction déjà calculée (`/predict` répond quand même, juste sans logging).

--> Ce que ces données permettent : c'est la source pour `monitoring.py` (taux d'erreur, latence), `drift_analysis.py` (dérive des données), `dashboard.py` (visualisation), et le profiling de performance; stocker à la fois l'input, l'output et la latence était nécessaire pour permettre cette analyse a posteriori.

--> Adéquation aux exigences du monitoring : le choix de PostgreSQL répond à une contrainte du monitoring en production : une API sous trafic réel reçoit des requêtes concurrentes, chacune déclenchant une écriture dans `prediction_logs`. SQLite verrouille l'écriture (=un seul writer à la fois) et sous charge concurrente ça peut faire échouer des écritures, avec un risque de trous dans les données de monitoring. PostgreSQL gère nativement les écritures concurrentes (MVCC), ce qui est nécessaire pour que le monitoring reste fiable sous trafic réel.

* Contrepartie : le stockage étant best-effort, une panne de Postgres pendant du trafic réel se traduit par des trous dans les données de monitoring sur cette fenêtre, ie l'API reste disponible, mais l'observabilité se dégrade --> compromis délibéré entre disponibilité du service et complétude du monitoring

--> Collecte des logs applicatifs avec Fluentd : en plus de `prediction_logs` (écrit directement par l'API), les logs JSON émis sur stdout (`_log()` dans `src/api.py` : `model_loaded`, `auth_failed`, `validation_rejected`...) sont collectés par Fluentd (`Dockerfile.fluentd`, `fluentd/fluent.conf`) via le driver de logging Docker, puis écrits dans une table Postgres séparée (`application_logs`, cf. `postgres-init/01_application_logs.sql`). Les deux canaux sont volontairement distincts : `prediction_logs` est alimenté directement par l'API pour l'analyse structurée (drift, latence), `application_logs` capture tous les événements applicatifs via un vrai pipeline de collecte de logs, découplé du code de l'API elle-même. Testé de bout en bout (`docker compose up`, appel réel, vérification `psql` que la ligne apparaît dans `application_logs` avec le bon `event` et `log_time`).

* Limites connues :
- Pas de politique de rétention : les logs s'accumulent indéfiniment, aucune purge automatique.
- Pas de chiffrement au repos : configuré par défaut sur le conteneur Postgres local.
- RGPD : ce projet tourne sur des données anonymisées (dataset Kaggle) à des fins pédagogiques. Si ce système traitait un jour de vraies données de clients, il faudrait ajouter chiffrement au repos, rétention limitée dans le temps, anonymisation/pseudonymisation des champs sensibles, et un mécanisme de droit à l'effacement.

## Optimisation des performances d'inférence

* Méthodologie : partir des données de monitoring. `src/profile_inference.py` commence par interroger `monitoring.py` sur les appels enregistrés en base, et récupère un enregistrement `input_json` pour profiler sur une requête authentique. Sur du trafic réel généré via l'API : latence moyenne 0,71ms / p95 1,09ms observées : c'est cette baseline et le fait qu'elle laissait deviner un potentiel d'optimisation vu l'écart avec les temps de predict mesurés par ailleurs, qui a motivé le profiling détaillé ci-dessous:

Un profiling (`uv run python src/profile_inference.py`) a décomposé `/predict` en étapes chronométrées séparément, sur le modèle et un enregistrement :

| Étape | Moyenne | p95 |
|---|---|---|
| Construction du DataFrame | 2,30ms | 2,46ms |
| Predict via wrapper `mlflow.pyfunc` (ancien chemin) | 5,27ms | 5,86ms |
| Predict LightGBM natif (sans `mlflow.pyfunc`) | 1,27ms | 1,84ms |

Constat fait : le calcul du modèle lui-même ne prend que 1,27ms, mais le wrapper `mlflow.pyfunc` représentait 75,9% du temps de predict.

Un profiling outillé plus fin (`uv run python src/profile_cprofile.py`, via `cProfile`) localise la source de cet overhead : sur 200 appels, le chemin `mlflow.pyfunc` déclenche 20,3 millions d'appels de fonction Python, contre 1,3 million pour le modèle natif. Le vrai goulot est la fonction `_enforce_named_col_schema` de mlflow, qui appelle `pandas.Index.union()` une fois /colonne du schéma. Ça revient à répéter cette opération environ 710 fois par requête (une fois par colonne du modèle) donc plus le modèle a de features, plus cette vérification de schéma coûte cher.

* Utilisation CPU (`psutil`, mesurée sur une charge soutenue de 2s, pas sur une requête isolée car trop courte pour une lecture stable) : env. 101% pour le chemin `mlflow.pyfunc`, env. 105% pour le natif (1 cœur = 100%, machine à 10 cœurs logiques). Les deux saturent un seul cœur pendant leur exécution : le CPU% seul ne révèle donc pas l'écart de perf entre les deux chemins ; c'est le temps CPU total consommé par requête (corrélé à la latence, pas au taux d'utilisation instantané) qui compte ici. Le goulot n'est pas un manque de ressources CPU disponibles, c'est un excès de travail Python inutile par requête (cf. `cProfile` ci-dessus).

* Optimisation appliquée : `src/api.py` charge désormais le modèle LightGBM natif (`mlflow.lightgbm.load_model`) au lieu du wrapper `mlflow.pyfunc`: même modèle, mêmes poids donc aucune perte de précision. Mesuré en conditions réelles sur `/predict` (`uv run python src/benchmark_predict.py`, 100 requêtes HTTP, même payload, avant/après) :

| | Moyenne | Médiane | p95 |
|---|---|---|---|
| Avant (wrapper `mlflow.pyfunc`) | 18,68ms | 15,03ms | 73,47ms |
| Après (modèle natif) | 6,42ms | 6,35ms | 6,97ms |

Résultat : env 3x plus rapide en moyenne, et la variance en queue de distribution (p95) est passée de + de 73ms à moins de 7ms.

Pour reproduire la mesure "avant" : `benchmark_predict.py` mesure `/predict` tel qu'il existe dans le code actuel (donc la version optimisée). Le commit `38a1613` (juste avant cette optimisation) contient encore l'ancien code base sur `mlflow.pyfunc` :

    git checkout 38a1613 -- src/api.py
    # relancer l'API, puis uv run python src/benchmark_predict.py
    git checkout HEAD -- src/api.py

* Deuxième round d'optimisation : le profiling initial montrait que la construction du `pd.DataFrame` (2,42ms) coûtait à elle seule plus que le calcul natif du modèle (1,13ms). En contournant aussi pandas et la couche `LGBMClassifier.predict_proba` ; appel direct de `model.booster_.predict()` sur un tableau numpy construit à la main, dans l'ordre exact de `model.feature_name_` ; la comparaison isolée montre un gain supplémentaire d'env 15x (`predict_proba(df)` : 1,09ms vs `booster_.predict(array)` : 0,07ms, résultats numériquement identiques, vérifié avec `np.allclose`).

Mesuré en conditions réelles sur `/predict` complet :

| | Moyenne | Médiane | p95 |
|---|---|---|---|
| Round 1 (modèle natif via `predict_proba(df)`) | 6,42ms | 6,35ms | 6,97ms |
| Round 2 (`booster_.predict` sur tableau numpy) | 3,67ms | 3,57ms | 4,42ms |

Au total depuis le point de départ : 18,68ms --> 3,67ms, soit environ 5x plus rapide!

* Impact sur la précision : aucun. 

- Round 1 (bypass `mlflow.pyfunc`) : le modèle natif chargé (`lightgbm-credit-scoring`) et le wrapper pyfunc (`lightgbm-credit-scoring-serving`) proviennent du même run d'entraînement (vérifié via les `run_id` MLflow), donc des mêmes poids.
- Round 2 (bypass pandas/sklearn) : `model.booster_.predict()` appelle directement le même booster que `LGBMClassifier.predict_proba()` en interne, sans transformation intermédiaire des valeurs.
- Validation numérique (`tests/test_no_regression.py`, automatisée en CI) : comparaison des probabilités retournées par l'ancien chemin (`mlflow.pyfunc`) et le nouveau (`booster_.predict`) sur 30 échantillons aux caractéristiques variées / écart maximal observé : 0.0 (tolérance testée : 1e-9). Puisque les probabilités sont identiques, toutes les métriques business qui en dérivent (AUC, recall, coût métier, seuil optimal) le sont aussi mécaniquement donc inutile de les recalculer.

* Optimisations envisagées mais non retenues :
  * Réduire la complexité du modèle aurait dégradé le coût métier déjà optimisé pour un gain marginal, vu que le calcul natif est déjà sous 1,5ms : ça a été estimé non justifié.
  * ONNX Runtime, testé empiriquement (`uv run python src/benchmark_onnx.py`, conversion via `onnxmltools`) plutôt que rejeté sur simple estimation : sur le calcul du modèle isolé, ONNX est réellement 90,4% plus rapide que `booster_.predict()` (0,085ms --> 0,008ms), précision quasi identique (écart 1,24e-08, dû au passage en float32). Mais intégré dans l'API et rebenchmarké de bout en bout sur `/predict` réel (même méthodologie que les rounds 1 et 2) : 3,67ms --> 3,55-3,62ms, un gain non significatif (dans la marge de bruit de mesure). Le calcul du modèle ne représentait déjà qu'une fraction infime des 3,67ms totaux (dominés par la validation métier, la construction du tableau, le logging) : gagner 90% sur une étape qui ne pèse déjà presque rien, ça ne change quasiment rien au temps de réponse total. Non retenu : le coût (nouvelle dépendance lourde `onnxruntime`/`onnxmltools`, conversion du modèle à chaque démarrage) dépasse le bénéfice réel mesuré.

## Justification de la configuration finale

* Modèle et inférence (software) : LightGBM natif, chargé directement via `mlflow.lightgbm` plutôt que par le wrapper générique `mlflow.pyfunc`, avec appel direct au Booster sur un tableau numpy plutôt que `LGBMClassifier.predict_proba(df)`. Une conversion ONNX a été testée empiriquement mais non retenue.

* Hardware : CPU uniquement, pas de GPU : LightGBM est un ensemble d'arbres de décision, pas un réseau de neurones : l'inférence sur CPU est déjà optimale pour ce type de modèle. Avec un modèle à 398 arbres inférant en env 1ms sur un CPU standard, un GPU n'apporterait aucun bénéfice pour ce cas d'usage et ajouterait un coût d'infrastructure et de complexité de déploiement injustifiés.

* API : FastAPI + Uvicorn : Choisi pour la documentation Swagger générée automatiquement, la validation de schéma native (Pydantic), et le support asynchrone.

* Tracking MLflow : SQLite : zéro administration, suffisant pour le volume d'un PoC local.

* Logs de production (`prediction_logs`) : PostgreSQL, dans un conteneur dédié (`docker-compose.yml` en local, service dédié en CI). stockage best-effort côté API

* Dashboard : Streamlit plutôt que Dash : Développement plus rapide pour un dashboard de monitoring interne aux besoins simples.

## Dashboard de monitoring

    uv run streamlit run src/dashboard.py

Le dashboard est accessible à http://localhost:8501.

## Tests et couverture

Les tests unitaires et d'intégration se lancent avec :

    uv run pytest tests/ --cov --cov-report=term-missing

```
Name                    Stmts   Miss  Cover   Missing
-----------------------------------------------------
src/api.py                120     14    88%   43, 65-66, 71-72, 92-94, 140-141, 157-158, 204-205
src/drift_analysis.py      58     12    79%   16, 76-93
src/final_model.py         26      0   100%
src/monitoring.py          87     13    85%   42, 96-97, 143-153
src/storage.py             44      0   100%
-----------------------------------------------------
TOTAL                     335     39    88%
```

`src/api.py`, `src/storage.py`, `src/monitoring.py` et `src/drift_analysis.py` sont couverts par des tests unitaires et d'intégration (`tests/`) : routes de l'API avec un modèle factice injecté, stockage sur une base Postgres de test isolée, détection d'anomalies et de drift sur des jeux de données construits.

`src/final_model.py` exclut explicitement (`# pragma: no cover`) `load_data`, `train_and_register`, `_register_serving_wrapper` et le bloc `__main__` : ce pipeline d'entraînement et d'enregistrement MLflow lit des données lourdes (`data/train_engineered.csv`) et écrit dans le Model Registry. Le tester en continu demanderait soit de le ré-exécuter à chaque run de tests (coûteux et non déterministe), soit de le mocker intégralement, ce qui ne testerait plus que du câblage et non un vrai comportement. Ce code est donc validé manuellement plutôt que par des tests automatisés ; seules `cout_metier` et `seuil_optimal`, la logique métier critique, sont testées et comptent dans la couverture (100%).

Les scripts exécutés manuellement (`src/dashboard.py`, `src/benchmark_onnx.py`, `src/benchmark_predict.py`, `src/profile_cprofile.py`, `src/profile_inference.py`, `src/generate_demo_predictions.py`, `src/seed_demo_data.py`) sont exclus de la mesure (`omit` dans `pyproject.toml`) : ce sont des outils d'exploration/mesure/démo ponctuels, pas du code applicatif couvert par la suite de tests automatisée.

Le taux de couverture global de 88% reflète donc fidèlement le code applicatif réellement testé automatiquement, une fois écarté ce qui est intentionnellement validé autrement.

## Dépendances

Les dépendances sont gérées avec uv. Pour installer l'environnement :

    uv sync
