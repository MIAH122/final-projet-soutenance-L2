# RiceInsight — version locale 2004–2026

Cette version utilise l'historique disponible dans la base WFP pour la série `Rice (local)` sur la période 2004–2026.

## Lancer en local

Depuis ce dossier :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Créer un fichier `.env` à partir de `.env.example` et renseigner vos identifiants.

Puis :

```bash
streamlit run app.py
```

## Modèle

Random Forest réentraîné sur les données 2004–2026 après création des variables temporelles et des variables retardées.

- 5 336 observations après préparation
- 4 268 entraînement / 1 068 test
- R² = 0.9655
- MAE = 112.24 Ar
- RMSE = 161.12 Ar
- split aléatoire 80/20, random_state=42

## Important

La source WFP contient `priceflag=actual` pour la période 2004–2019 et `priceflag=aggregate` pour les observations récentes disponibles jusqu'en 2026. La version finale conserve ce statut dans les données et utilise les deux ensembles pour couvrir 2004–2026.
