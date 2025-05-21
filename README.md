# Prédiction du Prix Bitcoin basée sur Ethereum

Ce projet utilise un modèle deep learning (CNN-BiLSTM) pour prédire le prix du Bitcoin en utilisant les données historiques d'Ethereum comme features principales.

## Structure du Projet

```
ETH-to-BTC/
├── README.md               # Ce fichier
├── requirements.txt        # Dépendances Python
├── setup.py                # Configuration pour l'installation
├── predict.py              # Script principal pour la prédiction
├── data/                   # Répertoire pour les données
├── models/                 # Répertoire pour les modèles sauvegardés
├── notebooks/              # Notebooks Jupyter pour l'exploration
└── src/                    # Code source principal
    ├── __init__.py
    ├── data/               # Module pour la collecte des données
    │   ├── __init__.py
    │   └── collector.py
    ├── features/           # Module pour le prétraitement
    │   ├── __init__.py
    │   └── preprocessing.py
    ├── models/             # Module pour les modèles
    │   ├── __init__.py
    │   └── cnn_bilstm.py
    ├── utils/              # Fonctions utilitaires
    │   ├── __init__.py
    │   └── visualization.py
    └── train.py            # Script d'entraînement
```

## Installation

1. Clonez le dépôt:
   ```bash
   git clone https://github.com/YoussefAIDT/ETH-to-BTC.git
   ```

2. Créez un environnement virtuel et installez les dépendances:
   ```bash
   python -m venv venv
   source venv/bin/activate  # ou venv\Scripts\activate sous Windows
   pip install -r requirements.txt
   ```

3. (Optionnel) Installez le package en mode développement:
   ```bash
   pip install -e .
   ```

## Utilisation

### Entraînement du modèle

Pour entraîner un nouveau modèle:

```bash
python -m src.train --days 730 --seq_length 30 --epochs 100 --predict_future
```

Options:
- `--days`: Nombre de jours de données historiques à récupérer (défaut: 730)
- `--seq_length`: Longueur des séquences temporelles (défaut: 30)
- `--train_ratio`: Proportion des données pour l'entraînement (défaut: 0.7)
- `--val_ratio`: Proportion des données pour la validation (défaut: 0.15)
- `--epochs`: Nombre maximal d'époques pour l'entraînement (défaut: 100)
- `--batch_size`: Taille des batchs pour l'entraînement (défaut: 32)
- `--model_name`: Nom du fichier pour sauvegarder le modèle (défaut: model_lstm_bitcoin_eth.h5)
- `--predict_future`: Activer la prédiction future (défaut: False)
- `--days_ahead`: Nombre de jours à prédire (défaut: 30)

### Prédiction avec un modèle existant

Pour faire des prédictions avec un modèle déjà entraîné:

```bash
python predict.py --model_path models/model_lstm_bitcoin_eth.h5 --days_ahead 30
```

Options:
- `--model_path`: Chemin vers le modèle sauvegardé (défaut: models/model_lstm_bitcoin_eth.h5)
- `--days`: Nombre de jours de données historiques à récupérer (défaut: 60)
- `--seq_length`: Longueur des séquences temporelles (défaut: 30)
- `--days_ahead`: Nombre de jours à prédire (défaut: 30)

## Description du Modèle

Le modèle est un réseau hybride CNN-BiLSTM qui:
1. Utilise des convolutions 1D pour extraire les caractéristiques des séquences temporelles
2. Utilise des couches LSTM bidirectionnelles pour capturer les dépendances à long terme
3. Applique des techniques de régularisation (dropout, régularisation L1-L2) pour éviter le surapprentissage
4. Utilise un mécanisme de correction de biais pour améliorer la précision des prédictions

## Features Utilisées

Le modèle utilise les features suivantes d'Ethereum pour prédire le prix du Bitcoin:
- Prix de clôture
- Moyennes mobiles (7, 14, 30 jours)
- Volatilité sur 7 jours
- Amplitude quotidienne
- Ratio volume/prix
- Rate of Change (5 et 10 jours)
- Indicateurs de momentum (5 et 10 jours)
- Rendements quotidiens

